"""
COCO Evaluator Module (benchmark.py)

Standalone benchmarking script that iterates through a local directory of 50 COCO dataset images,
processes each image through the agentic vision wrapper (vision_agent.py), transforms
the structured object predictions into standard COCO bounding box format, and evaluates
Mean Average Precision (mAP) utilizing pycocotools.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from audit_trail import log_interaction
from vision_agent import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_SLM_MODEL,
    analyze_image_with_gemini,
    analyze_image_with_hybrid,
    analyze_image_with_slm,
)

# Configure module logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("COCOBenchmark")

# Common aliases mapping natural language terms to standard 80 COCO categories
COCO_LABEL_ALIASES = {
    "sofa": "couch",
    "table": "dining table",
    "phone": "cell phone",
    "mobile": "cell phone",
    "smartphone": "cell phone",
    "tv screen": "tv",
    "television": "tv",
    "monitor": "tv",
    "pc": "laptop",
    "automobile": "car",
    "cab": "car",
    "taxi": "car",
    "pedestrian": "person",
    "man": "person",
    "woman": "person",
    "child": "person",
    "bicycler": "bicycle",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "plane": "airplane",
    "glass": "wine glass",
    "coffee cup": "cup",
    "mug": "cup",
    "kitchen & dining room table": "dining table",
}


def build_coco_category_map(coco_gt) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds case-insensitive name-to-ID and ID-to-name lookup dictionaries
    from the loaded COCO ground truth instance.
    """
    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}

    for cat in cats:
        c_id = cat["id"]
        c_name = cat["name"].lower().strip()
        name_to_id[c_name] = c_id
        id_to_name[c_id] = cat["name"]

    return name_to_id, id_to_name


def map_label_to_coco_id(label: str, name_to_id: Dict[str, int]) -> Optional[int]:
    """
    Maps an arbitrary predicted object label to the matching official COCO category ID.
    Supports lowercasing, alias substitution, and partial matching.
    """
    clean_label = label.lower().strip()

    # Direct match
    if clean_label in name_to_id:
        return name_to_id[clean_label]

    # Alias match
    if clean_label in COCO_LABEL_ALIASES:
        target = COCO_LABEL_ALIASES[clean_label]
        if target in name_to_id:
            return name_to_id[target]

    # Whole-word boundary match (avoids false positives like 'car' in 'cart')
    for coco_name, c_id in name_to_id.items():
        if re.search(rf"\b{re.escape(coco_name)}\b", clean_label):
            return c_id

    return None


def resolve_image_id(
    image_path: Path,
    coco_gt
) -> Optional[int]:
    """
    Determines the official COCO integer image ID corresponding to a given image file.
    Checks filename in ground truth metadata, or extracts integer from stem.
    """
    filename = image_path.name
    # Check if filename is directly registered in COCO
    for img_id, info in coco_gt.imgs.items():
        if info.get("file_name") == filename:
            return img_id

    # Fallback: COCO standard filenames are 12 digits, e.g. 000000000139.jpg -> 139
    stem = image_path.stem
    # Strip any common prefixes like 'COCO_val2014_' or 'COCO_val2017_'
    if "_" in stem:
        stem = stem.split("_")[-1]

    try:
        candidate_id = int(stem)
        if candidate_id in coco_gt.imgs:
            return candidate_id
    except ValueError:
        pass

    return None


def convert_gemini_box_to_coco_bbox(
    box_2d: List[int],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Converts Gemini normalized bounding box [ymin, xmin, ymax, xmax] (0 to 1000 scale)
    to COCO format [x, y, width, height] in pixel coordinates.
    """
    ymin, xmin, ymax, xmax = box_2d

    # Convert normalized 0-1000 to pixel dimensions
    x = (xmin / 1000.0) * img_width
    y = (ymin / 1000.0) * img_height
    w = ((xmax - xmin) / 1000.0) * img_width
    h = ((ymax - ymin) / 1000.0) * img_height

    # Clip values to image bounds
    x = max(0.0, min(float(img_width), x))
    y = max(0.0, min(float(img_height), y))
    w = max(0.0, min(float(img_width - x), w))
    h = max(0.0, min(float(img_height - y), h))

    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


def run_coco_benchmark(
    images_dir: Path,
    annotations_file: Path,
    limit: int = 50,
    api_key: Optional[str] = None,
    model_name: Optional[str] = "gemini-3.5-flash",
    backend: str = "hybrid",
    slm_model: str = DEFAULT_SLM_MODEL,
    detector_model: str = DEFAULT_DETECTOR_MODEL,
    conf_threshold: float = 0.05,
    output_detections_file: str = "benchmark_detections.json",
    delay_between_calls: float = 1.0
) -> Dict[str, Any]:
    """
    Executes the end-to-end COCO benchmark across up to `limit` images.

    Parameters:
        images_dir (Path): Local directory containing COCO images.
        annotations_file (Path): Path to COCO ground truth instances JSON file.
        limit (int): Number of images to evaluate (default: 50).
        api_key (str, optional): Gemini API Key.
        output_detections_file (str): Destination file for generated COCO predictions.
        delay_between_calls (float): Sleep seconds between API calls to avoid rate limits.

    Returns:
        dict: Detailed benchmark results including AP metrics and per-image stats.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise ImportError(
            "The 'pycocotools' package is required. Install it using 'pip install pycocotools'."
        ) from e

    logger.info(f"Loading COCO Ground Truth from: {annotations_file}")
    coco_gt = COCO(str(annotations_file))
    name_to_id, id_to_name = build_coco_category_map(coco_gt)

    # Collect images
    image_files = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not image_files:
        raise FileNotFoundError(f"No valid images found in directory: {images_dir}")

    eval_files = image_files[:limit]
    logger.info(f"Starting benchmark evaluation on {len(eval_files)} images (limit={limit}, backend={backend})...")

    coco_detections: List[Dict[str, Any]] = []
    evaluated_image_ids: Set[int] = set()
    skipped_images: List[str] = []

    for index, img_path in enumerate(eval_files, start=1):
        img_id = resolve_image_id(img_path, coco_gt)
        if img_id is None:
            logger.warning(f"[{index}/{len(eval_files)}] Could not resolve COCO ID for {img_path.name}. Skipping.")
            skipped_images.append(img_path.name)
            continue

        evaluated_image_ids.add(img_id)

        # Read image to obtain dimensions
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        logger.info(f"[{index}/{len(eval_files)}] Processing {img_path.name} (ID: {img_id}, {img_w}x{img_h})...")

        try:
            # 1. Process through Vision Agent
            if backend == "hybrid":
                vision_result = analyze_image_with_hybrid(
                    image_input=img_path,
                    detector_model_id=detector_model,
                    caption_model_id=slm_model,
                    include_caption=False,
                    confidence_threshold=conf_threshold,
                )
            elif backend == "slm":
                vision_result = analyze_image_with_slm(
                    image_input=img_path,
                    model_id=slm_model,
                    include_caption=False,
                )
            else:
                vision_result = analyze_image_with_gemini(
                    image_input=img_path,
                    api_key=api_key,
                    model_name=model_name
                )

            # 2. Maintain strict audit trail
            log_interaction(
                image_identifier=img_path.name,
                gemini_output=vision_result,
                metadata={
                    "benchmark_run": True,
                    "coco_image_id": img_id,
                    "backend": backend,
                }
            )

            # 3. Parse predicted objects into COCO format
            detected_objs = vision_result.get("detected_objects", [])
            for obj in detected_objs:
                label = obj.get("label", "")
                conf = float(obj.get("confidence", 0.0))
                box_2d = obj.get("box_2d", [0, 0, 1000, 1000])

                category_id = map_label_to_coco_id(label, name_to_id)
                if category_id is None:
                    # Predicted class is not in COCO categories (e.g. door, wall)
                    continue

                coco_bbox = convert_gemini_box_to_coco_bbox(box_2d, img_w, img_h)

                coco_detections.append({
                    "image_id": int(img_id),
                    "category_id": int(category_id),
                    "bbox": coco_bbox,
                    "score": round(conf, 4)
                })

        except Exception as err:
            logger.error(f"Error processing {img_path.name}: {err}")

        # Rate limit safety delay (only needed for remote Gemini API)
        if backend == "gemini" and index < len(eval_files):
            time.sleep(delay_between_calls)

    logger.info(f"Total detections generated across {len(evaluated_image_ids)} images: {len(coco_detections)}")

    # Write detections file
    out_path = Path(output_detections_file)
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(coco_detections, out_f, indent=2)
    logger.info(f"Saved COCO predictions to: {out_path}")

    # Run pycocotools evaluation
    metrics_summary: Dict[str, Any] = {
        "num_images_evaluated": len(evaluated_image_ids),
        "total_detections": len(coco_detections),
        "skipped_images": skipped_images
    }

    if not coco_detections:
        logger.warning("No COCO-compatible detections were generated. Evaluation skipped.")
        return metrics_summary

    logger.info("Initializing pycocotools COCOeval...")
    coco_dt = coco_gt.loadRes(str(out_path))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = list(evaluated_image_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract standard COCO metrics
    stats = coco_eval.stats
    metrics_summary.update({
        "mAP_IoU_0.50_0.95": float(stats[0]),
        "mAP_IoU_0.50": float(stats[1]),
        "mAP_IoU_0.75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_maxDets_1": float(stats[6]),
        "AR_maxDets_10": float(stats[7]),
        "AR_maxDets_100": float(stats[8])
    })

    return metrics_summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark Vision Agent (Hybrid Local, Florence-2 SLM, or Gemini) on COCO Images with pycocotools."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/coco_val2017_50/images",
        help="Path to directory containing COCO images."
    )
    parser.add_argument(
        "--annotations-file",
        type=str,
        default="data/coco_val2017_50/instances_val2017_subset.json",
        help="Path to COCO instance annotations JSON file."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of images to evaluate (default: 50)."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hybrid", "slm", "gemini"],
        default="hybrid",
        help="Inference backend: 'hybrid' (RT-DETR + Florence-2), 'slm' (Florence-2), or 'gemini'."
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default=DEFAULT_DETECTOR_MODEL,
        help=f"Local HuggingFace detector ID for hybrid mode (default: {DEFAULT_DETECTOR_MODEL})."
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.05,
        help="Confidence threshold for hybrid detector evaluation (default: 0.05 for COCO PR curve)."
    )
    parser.add_argument(
        "--slm-model",
        type=str,
        default=DEFAULT_SLM_MODEL,
        help=f"Local HuggingFace SLM identifier (default: {DEFAULT_SLM_MODEL})."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google Gemini API Key (falls back to GEMINI_API_KEY env var)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash",
        help="Gemini model identifier (default: gemini-3.5-flash)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_detections.json",
        help="Output JSON file for generated predictions."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    images_path = Path(args.images_dir)
    annotations_path = Path(args.annotations_file)

    if not images_path.exists() or not annotations_path.exists():
        raise FileNotFoundError(
            f"COCO dataset files not found ({images_path} or {annotations_path}). "
            "Run 'python prepare_coco_50.py --output-dir data/coco_val2017_50 --limit 50' "
            "or pass --images-dir and --annotations-file."
        )

    if args.backend == "hybrid":
        mode_desc = f"Hybrid Local ({args.detector_model}, thr={args.conf_threshold})"
    elif args.backend == "slm":
        mode_desc = f"Local SLM ({args.slm_model})"
    else:
        mode_desc = f"Live Gemini ({args.model})"

    print("=" * 60)
    print("       ASSISTIVE AI AGENT - COCO EVALUATOR BENCHMARK       ")
    print("=" * 60)
    print(f" Images Directory : {images_path}")
    print(f" Annotations File : {annotations_path}")
    print(f" Image Limit      : {args.limit}")
    print(f" Execution Mode   : {mode_desc}")
    print("=" * 60)

    results = run_coco_benchmark(
        images_dir=images_path,
        annotations_file=annotations_path,
        limit=args.limit,
        api_key=args.api_key,
        model_name=args.model,
        backend=args.backend,
        slm_model=args.slm_model,
        detector_model=args.detector_model,
        conf_threshold=args.conf_threshold,
        output_detections_file=args.output
    )

    print("\n" + "=" * 60)
    print("                 BENCHMARK SUMMARY RESULTS                 ")
    print("=" * 60)
    print(f" Images Evaluated : {results.get('num_images_evaluated')}")
    print(f" Total Detections : {results.get('total_detections')}")
    if "mAP_IoU_0.50_0.95" in results:
        print(f" mAP @ 0.50:0.95  : {results.get('mAP_IoU_0.50_0.95'):.4f}")
        print(f" mAP @ 0.50       : {results.get('mAP_IoU_0.50'):.4f}")
        print(f" mAP @ 0.75       : {results.get('mAP_IoU_0.75'):.4f}")
        print(f" mAP (Small)      : {results.get('mAP_small'):.4f}")
        print(f" mAP (Medium)     : {results.get('mAP_medium'):.4f}")
        print(f" mAP (Large)      : {results.get('mAP_large'):.4f}")
    print("=" * 60)

    # Save summary report
    summary_path = Path("benchmark_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(results, sf, indent=2)
    print(f"Metrics summary written to: {summary_path.resolve()}\n")


if __name__ == "__main__":
    main()
