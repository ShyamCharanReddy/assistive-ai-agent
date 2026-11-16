"""
Agentic Vision Wrapper for Assistive AI Agent.

Interfaces directly with the Google Gemini API to process visual inputs
and generate structured JSON guidance for visually impaired users.
Features dynamic model discovery, multi-model fallback resiliency,
and robust JSON parsing.
"""

import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

# Configure module logger
logger = logging.getLogger("VisionAgent")

# Prioritized list of vision-capable models (modern 3.5/3.8 Flash, latest aliases, and fallbacks)
FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest"
]

DEFAULT_MODEL_NAME = "gemini-3.5-flash"
DEFAULT_SLM_MODEL = "microsoft/Florence-2-base-ft"
DEFAULT_DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"

# Cached discovered model name to avoid repeated discovery calls
_CACHED_ACTIVE_MODEL: Optional[str] = None
# Cached local models keyed by HuggingFace model ID
_FLORENCE_CACHE: Dict[str, Any] = {}
_RTDETR_CACHE: Dict[str, Any] = {}

NAVIGATIONAL_HAZARDS = {
    "car", "truck", "bus", "motorcycle", "bicycle", "train",
    "fire hydrant", "stop sign", "stairs", "dog", "skateboard"
}

# System Prompt crafted for Assistive Vision & Navigation
SYSTEM_INSTRUCTION = """
You are an expert Assistive AI Vision Agent designed to aid visually impaired individuals navigate their environment safely and independently.
Your goal is to visually analyze the provided image with high precision, spatial reasoning, and hazard awareness.

You must respond ONLY with a valid JSON object matching the exact schema below, with no conversational filler or markdown wrappers:

{
  "summary": "A concise, natural, and descriptive spoken summary of the scene tailored for auditory navigation. Mention key spatial relationships (e.g. 'At 12 o'clock, 2 meters ahead...'), ground clearance, pathway availability, and immediate surroundings.",
  "detected_objects": [
    {
      "label": "standardized object name (e.g. person, chair, table, car, bicycle, stairs, door, cup)",
      "confidence": 0.95,
      "box_2d": [ymin, xmin, ymax, xmax], // Normalized coordinates from 0 to 1000 representing bounding box
      "description": "Short description of object position, posture, or orientation"
    }
  ],
  "reasoning_and_confidence": {
    "scene_type": "indoor / outdoor / sidewalk / hallway / room / crosswalk",
    "lighting_condition": "bright daylight / dim / well-lit indoor / shadows / glare",
    "primary_hazards": [
      "List of any detected hazards, drop-offs, stairs, approaching vehicles, or tripping obstacles"
    ],
    "overall_confidence": 0.94,
    "confidence_metrics": {
      "detection_certainty": "high / medium / low",
      "spatial_accuracy": "high / medium / low",
      "lighting_clarity": "clear / degraded / shadowed"
    },
    "reasoning_notes": "Step-by-step reasoning on safe navigation corridors, obstacles, and navigational recommendations."
  }
}
"""


def _load_image(image_input: Union[str, Path, Image.Image, bytes, io.BytesIO]) -> Image.Image:
    """Standardizes varied image input types into a PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        return Image.open(img_path).convert("RGB")
    elif isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, io.BytesIO):
        image_input.seek(0)
        return Image.open(image_input).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def _clean_json_markdown(raw_text: str) -> str:
    """Strips markdown code fence wrappers (```json ... ```) if present."""
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def _validate_and_sanitize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validates that required keys are present and populates defaults if missing."""
    sanitized: Dict[str, Any] = {
        "summary": data.get("summary", "A scene has been detected, but no detailed summary was provided."),
        "detected_objects": [],
        "reasoning_and_confidence": {}
    }

    raw_objects = data.get("detected_objects", [])
    if isinstance(raw_objects, list):
        for obj in raw_objects:
            if isinstance(obj, dict):
                label = str(obj.get("label", "unknown")).strip()
                conf = float(obj.get("confidence", 0.8))
                conf = max(0.0, min(1.0, conf))
                box = obj.get("box_2d", [0, 0, 1000, 1000])
                if not (isinstance(box, list) and len(box) == 4):
                    box = [0, 0, 1000, 1000]
                sanitized["detected_objects"].append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "box_2d": [int(coord) for coord in box],
                    "description": str(obj.get("description", ""))
                })

    raw_reasoning = data.get("reasoning_and_confidence", {})
    if isinstance(raw_reasoning, dict):
        sanitized["reasoning_and_confidence"] = {
            "scene_type": raw_reasoning.get("scene_type", "unspecified"),
            "lighting_condition": raw_reasoning.get("lighting_condition", "normal"),
            "primary_hazards": raw_reasoning.get("primary_hazards", []),
            "overall_confidence": float(raw_reasoning.get("overall_confidence", 0.85)),
            "confidence_metrics": raw_reasoning.get("confidence_metrics", {
                "detection_certainty": "medium",
                "spatial_accuracy": "medium",
                "lighting_clarity": "clear"
            }),
            "reasoning_notes": raw_reasoning.get("reasoning_notes", "Scene analyzed successfully.")
        }

    return sanitized


def get_available_vision_model(api_key: str) -> str:
    """
    Dynamically identifies an active, supported Gemini vision model for the API key.
    Falls back to prioritized list if listing models fails or times out.
    """
    global _CACHED_ACTIVE_MODEL
    if _CACHED_ACTIVE_MODEL:
        return _CACHED_ACTIVE_MODEL

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key, transport="rest")
        available_models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        # Find first matching supported candidate
        for candidate in FALLBACK_MODELS:
            if candidate in available_models:
                logger.info(f"Dynamically discovered active vision model: {candidate}")
                _CACHED_ACTIVE_MODEL = candidate
                return candidate
    except Exception as e:
        logger.warning(f"Could not list remote models ({e}). Defaulting to prioritized list.")

    # Default to top priority candidate
    _CACHED_ACTIVE_MODEL = FALLBACK_MODELS[0]
    return _CACHED_ACTIVE_MODEL


def analyze_image_with_gemini(
    image_input: Union[str, Path, Image.Image, bytes, io.BytesIO],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sends an image to Gemini API with automatic model fallback and structured JSON parsing.

    Parameters:
        image_input: File path, PIL Image, or raw byte stream.
        api_key (str, optional): Gemini API Key.
        model_name (str, optional): Specific model identifier to try first.

    Returns:
        dict: Validated structured dictionary containing 'summary', 'detected_objects', and 'reasoning_and_confidence'.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Gemini API key is required. Set the GEMINI_API_KEY environment variable "
            "or pass api_key to analyze_image_with_gemini()."
        )

    pil_image = _load_image(image_input)

    try:
        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "The 'google-generativeai' package is not installed. Run 'pip install google-generativeai'."
        ) from e

    # Configure REST transport for stable HTTPS connectivity
    genai.configure(api_key=resolved_key, transport="rest")

    # Build sequence of models to attempt: requested model first, then fallbacks
    models_to_try: List[str] = []
    if model_name:
        models_to_try.append(model_name)
    # Add candidate models without duplicates
    for cand in FALLBACK_MODELS:
        if cand not in models_to_try:
            models_to_try.append(cand)

    prompt = (
        "Analyze this image thoroughly for a visually impaired user. "
        "Identify all key objects, obstacles, clear paths, and immediate hazards. "
        "Return your analysis strictly in the requested structured JSON format."
    )

    last_error: Optional[Exception] = None

    for candidate in models_to_try:
        logger.info(f"Attempting vision analysis with Gemini model: {candidate}")
        try:
            model = genai.GenerativeModel(
                model_name=candidate,
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                }
            )

            response = model.generate_content([prompt, pil_image])
            raw_text = response.text

            # Parse and validate JSON
            cleaned_json = _clean_json_markdown(raw_text)
            parsed_dict = json.loads(cleaned_json)
            validated_result = _validate_and_sanitize_response(parsed_dict)

            # Update cache with working model
            global _CACHED_ACTIVE_MODEL
            _CACHED_ACTIVE_MODEL = candidate
            logger.info(f"Successfully processed image with model: {candidate}")
            return validated_result

        except Exception as e:
            last_error = e
            err_msg = str(e)
            logger.warning(f"Model '{candidate}' failed: {err_msg}. Attempting fallback...")

            # If it's an authentication error (e.g. invalid API key), fail fast
            if "API_KEY_INVALID" in err_msg or "403" in err_msg:
                raise RuntimeError(f"Authentication failed: Please check your GEMINI_API_KEY. Details: {e}") from e

            # Continue to next model in sequence
            continue

    # If all models in the fallback chain failed:
    raise RuntimeError(
        f"All attempted Gemini models failed. Last error: {last_error}. "
        "Please check your API key, network connection, or try again later."
    )


def _load_from_cache_or_hub(loader_fn, model_id: str, **kwargs):
    """Loads from local HuggingFace cache instantaneously first, downloading only if uncached."""
    try:
        return loader_fn(model_id, local_files_only=True, **kwargs)
    except Exception:
        return loader_fn(model_id, **kwargs)


def _load_florence_model(model_id: str = DEFAULT_SLM_MODEL):
    """Loads and caches a local Florence-2 model and processor on CPU/GPU."""
    if model_id in _FLORENCE_CACHE:
        return _FLORENCE_CACHE[model_id]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "Local SLM inference requires 'torch', 'transformers', 'einops', and 'timm'. "
            "Install them with: pip install torch transformers einops timm"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Loading local SLM '{model_id}' on {device}...")
    # attn_implementation="eager" avoids flash_attn issues on CPU environments.
    model = _load_from_cache_or_hub(
        AutoModelForCausalLM.from_pretrained,
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    processor = _load_from_cache_or_hub(
        AutoProcessor.from_pretrained,
        model_id,
        trust_remote_code=True,
    )
    model.eval()

    _FLORENCE_CACHE[model_id] = (model, processor, device, torch_dtype)
    return _FLORENCE_CACHE[model_id]


def _load_rtdetr_model(model_id: str = DEFAULT_DETECTOR_MODEL):
    """Loads and caches a local RT-DETR object detector and image processor."""
    if model_id in _RTDETR_CACHE:
        return _RTDETR_CACHE[model_id]

    try:
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    except ImportError as e:
        raise ImportError(
            "Local RT-DETR inference requires 'torch' and 'transformers>=4.46.0'."
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading local RT-DETR detector '{model_id}' on {device}...")
    processor = _load_from_cache_or_hub(RTDetrImageProcessor.from_pretrained, model_id)
    model = _load_from_cache_or_hub(RTDetrForObjectDetection.from_pretrained, model_id).to(device)
    model.eval()

    _RTDETR_CACHE[model_id] = (model, processor, device)
    return _RTDETR_CACHE[model_id]


def _run_florence_task(
    model,
    processor,
    device: str,
    torch_dtype,
    pil_image: Image.Image,
    task_prompt: str,
    num_beams: int = 1,
) -> Dict[str, Any]:
    """Runs a single Florence-2 prompt task (<OD> or <MORE_DETAILED_CAPTION>)."""
    import torch

    inputs = processor(text=task_prompt, images=pil_image, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device, torch_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=512,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=(num_beams > 1),
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(pil_image.width, pil_image.height),
    )


def _spatial_direction(box_2d: List[int]) -> str:
    """Converts normalized [ymin, xmin, ymax, xmax] center into clock orientation."""
    x_center = (box_2d[1] + box_2d[3]) / 2.0
    if x_center < 350:
        return "on the left (10 o'clock)"
    elif x_center > 650:
        return "on the right (2 o'clock)"
    return "straight ahead (12 o'clock)"


def _build_detection_payload(
    raw_detections: List[tuple],
    width: int,
    height: int,
    summary: str,
    scene_type: str,
    reasoning_notes: str,
    hazard_conf_cutoff: float = 0.35,
) -> Dict[str, Any]:
    """Converts (bbox_xyxy, label, confidence) tuples into the validated agent schema."""
    detected_objects: List[Dict[str, Any]] = []
    hazards: List[str] = []
    confident_scores: List[float] = []

    for bbox, label, score in raw_detections:
        x1, y1, x2, y2 = bbox
        ymin = max(0, min(1000, int(round((y1 / max(1, height)) * 1000))))
        xmin = max(0, min(1000, int(round((x1 / max(1, width)) * 1000))))
        ymax = max(0, min(1000, int(round((y2 / max(1, height)) * 1000))))
        xmax = max(0, min(1000, int(round((x2 / max(1, width)) * 1000))))
        box_2d = [ymin, xmin, ymax, xmax]

        clean_label = str(label).strip().lower()
        conf = round(max(0.0, min(1.0, float(score))), 4)
        direction = _spatial_direction(box_2d)

        detected_objects.append({
            "label": clean_label,
            "confidence": conf,
            "box_2d": box_2d,
            "description": f"{clean_label.title()} detected {direction}.",
        })

        if conf >= hazard_conf_cutoff:
            confident_scores.append(conf)
            if clean_label in NAVIGATIONAL_HAZARDS:
                hazard_desc = f"{clean_label} {direction}"
                if hazard_desc not in hazards:
                    hazards.append(hazard_desc)

    if confident_scores:
        overall_conf = round(max(confident_scores), 4)
        certainty = "high" if overall_conf >= 0.6 else "medium"
    else:
        overall_conf = 0.70
        certainty = "low"

    result = {
        "summary": summary or "Scene analyzed using local vision model.",
        "detected_objects": detected_objects,
        "reasoning_and_confidence": {
            "scene_type": scene_type,
            "lighting_condition": "assessed_locally",
            "primary_hazards": hazards,
            "overall_confidence": overall_conf,
            "confidence_metrics": {
                "detection_certainty": certainty,
                "spatial_accuracy": "high",
                "lighting_clarity": "clear",
            },
            "reasoning_notes": reasoning_notes,
        },
    }
    return _validate_and_sanitize_response(result)


def analyze_image_with_slm(
    image_input: Union[str, Path, Image.Image, bytes, io.BytesIO],
    model_id: str = DEFAULT_SLM_MODEL,
    include_caption: bool = True,
    num_beams: int = 1,
) -> Dict[str, Any]:
    """
    Runs local Vision SLM inference (Microsoft Florence-2) to produce structured
    object detections and scene descriptions matching the Gemini JSON schema.

    Parameters:
        image_input: File path, PIL Image, or raw bytes.
        model_id (str): HuggingFace model ID (default: 'microsoft/Florence-2-base-ft').
        include_caption (bool): Whether to run <MORE_DETAILED_CAPTION> in addition to <OD>.
        num_beams (int): Beam search width (1 is fastest on CPU; 3 improves recall slightly).
    """
    pil_image = _load_image(image_input)
    width, height = pil_image.size

    model, processor, device, torch_dtype = _load_florence_model(model_id)

    # 1. Object Detection (<OD>)
    od_raw = _run_florence_task(
        model, processor, device, torch_dtype, pil_image, "<OD>", num_beams=num_beams
    )
    od_data = od_raw.get("<OD>", {})
    raw_detections = [
        (bbox, label, 0.9)
        for bbox, label in zip(od_data.get("bboxes", []), od_data.get("labels", []))
    ]

    # 2. Detailed Scene Captioning (<MORE_DETAILED_CAPTION>)
    if include_caption:
        cap_raw = _run_florence_task(
            model, processor, device, torch_dtype, pil_image, "<MORE_DETAILED_CAPTION>", num_beams=num_beams
        )
        summary = str(cap_raw.get("<MORE_DETAILED_CAPTION>", "")).strip()
    else:
        obj_names = ", ".join(sorted({str(lbl).strip().lower() for _, lbl, _ in raw_detections}))
        summary = f"Detected objects: {obj_names}." if obj_names else "No objects detected."

    return _build_detection_payload(
        raw_detections=raw_detections,
        width=width,
        height=height,
        summary=summary,
        scene_type="local_slm_inference",
        reasoning_notes=(
            f"Analyzed locally with {model_id} on {device.upper()} ({width}x{height}). "
            f"Detected {len(raw_detections)} objects."
        ),
    )


def analyze_image_with_hybrid(
    image_input: Union[str, Path, Image.Image, bytes, io.BytesIO],
    detector_model_id: str = DEFAULT_DETECTOR_MODEL,
    caption_model_id: str = DEFAULT_SLM_MODEL,
    include_caption: bool = True,
    confidence_threshold: float = 0.35,
    max_detections: int = 100,
    num_beams: int = 1,
) -> Dict[str, Any]:
    """
    Hybrid local pipeline: pairs a fast calibrated COCO detector (RT-DETR)
    for accurate bounding boxes + scores with Florence-2 for natural scene narration.

    Parameters:
        image_input: File path, PIL Image, or raw bytes.
        detector_model_id (str): Local RT-DETR HuggingFace model ID.
        caption_model_id (str): Local Florence-2 HuggingFace model ID for captioning.
        include_caption (bool): Run Florence-2 <MORE_DETAILED_CAPTION> (False speeds up COCO benchmarks).
        confidence_threshold (float): Minimum RT-DETR score to retain.
        max_detections (int): Maximum detections sorted by descending confidence.
        num_beams (int): Beam search width for Florence-2 caption generation.
    """
    import torch

    pil_image = _load_image(image_input)
    width, height = pil_image.size

    det_model, det_processor, det_device = _load_rtdetr_model(detector_model_id)
    inputs = det_processor(images=pil_image, return_tensors="pt").to(det_device)
    with torch.no_grad():
        outputs = det_model(**inputs)

    target_sizes = torch.tensor([[height, width]], device=det_device)
    processed = det_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )[0]

    raw_detections = []
    for score, label_id, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
        label_name = det_model.config.id2label[int(label_id.item())]
        raw_detections.append((box.tolist(), label_name, float(score.item())))

    raw_detections.sort(key=lambda item: item[2], reverse=True)
    raw_detections = raw_detections[:max_detections]

    confident_cutoff = max(0.35, confidence_threshold)
    if include_caption:
        cap_model, cap_processor, cap_device, cap_dtype = _load_florence_model(caption_model_id)
        cap_raw = _run_florence_task(
            cap_model,
            cap_processor,
            cap_device,
            cap_dtype,
            pil_image,
            "<MORE_DETAILED_CAPTION>",
            num_beams=num_beams,
        )
        summary = str(cap_raw.get("<MORE_DETAILED_CAPTION>", "")).strip()
    else:
        obj_names = ", ".join(
            sorted({str(lbl).strip().lower() for _, lbl, s in raw_detections if s >= confident_cutoff})
        )
        summary = f"Detected objects: {obj_names}." if obj_names else "No high-confidence objects detected."

    return _build_detection_payload(
        raw_detections=raw_detections,
        width=width,
        height=height,
        summary=summary,
        scene_type="hybrid_rtdetr_florence2",
        reasoning_notes=(
            f"Hybrid local inference: detector={detector_model_id} ({det_device.upper()}), "
            f"captioner={caption_model_id if include_caption else 'skipped'}. "
            f"Retained {len(raw_detections)} detections (threshold={confidence_threshold})."
        ),
        hazard_conf_cutoff=confident_cutoff,
    )


if __name__ == "__main__":
    print("[VisionAgent] Default local hybrid pipeline:")
    print(f"  Detector : {DEFAULT_DETECTOR_MODEL}")
    print(f"  Captioner: {DEFAULT_SLM_MODEL}")
    print(f"  Gemini   : {DEFAULT_MODEL_NAME}")
