"""
Utility script to download official COCO val2017 annotations and a real N-image
subset (default: 50 images) without downloading the full 18 GB train2017 or
1 GB val2017 archive.
"""

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import requests

ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_JSON_MEMBER = "annotations/instances_val2017.json"


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Streams a remote file to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def prepare_coco_subset(output_dir: Path, limit: int = 50) -> None:
    """Downloads instances_val2017.json and `limit` real COCO val2017 images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    subset_anno_path = output_dir / "instances_val2017_subset.json"
    tmp_zip = Path("/tmp/annotations_trainval2017.zip")

    if not tmp_zip.exists():
        print(f"Downloading official COCO annotations zip from {ANNOTATIONS_ZIP_URL}...")
        download_file(ANNOTATIONS_ZIP_URL, tmp_zip)
        print(f"Downloaded {tmp_zip} ({tmp_zip.stat().st_size / (1024 * 1024):.1f} MB)")

    print(f"Extracting {VAL_JSON_MEMBER} from zip...")
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        with zf.open(VAL_JSON_MEMBER) as src:
            full_coco: Dict = json.load(src)

    # Remove large zip immediately to conserve disk space
    tmp_zip.unlink(missing_ok=True)

    # Select `limit` images that have at least one ground-truth annotation
    annotated_img_ids = {ann["image_id"] for ann in full_coco["annotations"]}
    selected_images: List[Dict] = []
    for img in full_coco["images"]:
        if img["id"] in annotated_img_ids:
            selected_images.append(img)
        if len(selected_images) >= limit:
            break

    selected_ids = {img["id"] for img in selected_images}
    selected_annotations = [
        ann for ann in full_coco["annotations"] if ann["image_id"] in selected_ids
    ]

    subset_coco = {
        "info": full_coco.get("info", {}),
        "licenses": full_coco.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": full_coco["categories"],
    }

    with open(subset_anno_path, "w", encoding="utf-8") as f:
        json.dump(subset_coco, f)
    print(f"Saved subset annotations ({len(selected_images)} images, {len(selected_annotations)} boxes) -> {subset_anno_path}")

    # Download each selected image
    for idx, img in enumerate(selected_images, start=1):
        img_dest = images_dir / img["file_name"]
        if img_dest.exists() and img_dest.stat().st_size > 0:
            continue
        url = img.get("coco_url") or f"http://images.cocodataset.org/val2017/{img['file_name']}"
        download_file(url, img_dest, chunk_size=65536)
        if idx % 10 == 0 or idx == len(selected_images):
            print(f"Downloaded {idx}/{len(selected_images)} real COCO images...")

    print("Real COCO subset preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare real COCO val2017 subset")
    parser.add_argument("--output-dir", type=str, default="data/coco_val2017_50")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    prepare_coco_subset(Path(args.output_dir), limit=args.limit)
