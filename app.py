"""
UI & Main Execution Module (app.py)

Streamlit-based user interface for the Assistive AI Agent.
Enables visually impaired users and caregivers to capture scenes via webcam,
upload images, or use hands-free VOICE TRIGGERS ('Capture', 'Describe', 'Capture and Describe').
Auto-routes spoken audio guidance to connected Bluetooth earphones and maintains
a local audit trail.
"""

import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

# Import core modules
from audit_trail import clear_audit_trail, load_audit_trail, log_interaction
from audio_utils import speak_vision_summary, text_to_speech
from vision_agent import (
    DEFAULT_DETECTOR_MODEL,
    DEFAULT_SLM_MODEL,
    analyze_image_with_gemini,
    analyze_image_with_hybrid,
    analyze_image_with_slm,
)
from voice_trigger import (
    VoiceCommand,
    get_command_acknowledgment,
    get_voice_controller_html,
    parse_audio_command_with_gemini,
    parse_voice_transcript,
)

# Streamlit Page Setup
st.set_page_config(
    page_title="Assistive Vision AI - Voice & Vision Agent",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Accessibility Styling
st.markdown("""
<style>
    .big-font { font-size: 20px !important; font-weight: 500; }
    .hazard-box {
        background-color: #ffebee;
        border-left: 6px solid #d32f2f;
        padding: 14px;
        margin-bottom: 15px;
        border-radius: 4px;
        color: #b71c1c;
        font-weight: 600;
    }
    .safe-box {
        background-color: #e8f5e9;
        border-left: 6px solid #2e7d32;
        padding: 14px;
        margin-bottom: 15px;
        border-radius: 4px;
        color: #1b5e20;
        font-weight: 600;
    }
    .summary-box {
        background-color: #e3f2fd;
        border-left: 6px solid #1976d2;
        padding: 16px;
        margin-bottom: 20px;
        border-radius: 6px;
        font-size: 1.15rem;
        line-height: 1.6;
        color: #0d47a1;
    }
    .stButton>button {
        height: 3em;
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def draw_bounding_boxes(image: Image.Image, detected_objects: List[Dict[str, Any]]) -> Image.Image:
    """Overlays 2D bounding boxes and labels onto the image for sighted companions."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    for idx, obj in enumerate(detected_objects):
        box = obj.get("box_2d", [0, 0, 1000, 1000])
        label = obj.get("label", "object")
        conf = obj.get("confidence", 0.0)

        ymin, xmin, ymax, xmax = box
        left = (xmin / 1000.0) * width
        top = (ymin / 1000.0) * height
        right = (xmax / 1000.0) * width
        bottom = (ymax / 1000.0) * height

        draw.rectangle([left, top, right, bottom], outline="#00e676", width=4)
        caption = f"{label.title()} ({int(conf * 100)}%)"
        tag_bg = [left, max(0, top - 24), left + (len(caption) * 9) + 8, top]
        draw.rectangle(tag_bg, fill="#00e676")
        draw.text((left + 4, max(0, top - 22)), caption, fill="black")

    return annotated


def process_analysis_pipeline(
    image: Image.Image,
    image_name: str,
    api_key: str,
    selected_model: str,
    tts_engine: str,
    play_system: bool,
    backend: str = "hybrid",
    confidence_threshold: float = 0.55,
) -> bool:
    """
    Core execution pipeline: Runs vision reasoning (Hybrid, Local SLM, or Gemini),
    updates audit trail, and plays audio guidance straight to Bluetooth earphones.
    """
    start_time = time.time()
    with st.spinner(f"Analyzing environment with {backend.upper()} ({selected_model})..."):
        try:
            if backend == "hybrid":
                vision_result = analyze_image_with_hybrid(
                    image_input=image,
                    detector_model_id=selected_model,
                    caption_model_id=DEFAULT_SLM_MODEL,
                    include_caption=True,
                    confidence_threshold=confidence_threshold,
                )
            elif backend == "slm":
                vision_result = analyze_image_with_slm(
                    image_input=image,
                    model_id=selected_model,
                    include_caption=True,
                )
            else:
                vision_result = analyze_image_with_gemini(
                    image_input=image,
                    api_key=api_key,
                    model_name=selected_model
                )

            latency_sec = round(time.time() - start_time, 2)

            # 1. Log to Audit Trail
            audit_entry = log_interaction(
                image_identifier=image_name,
                gemini_output=vision_result,
                metadata={
                    "latency_seconds": latency_sec,
                    "dimensions": f"{image.width}x{image.height}",
                    "backend": backend,
                    "mode": selected_model
                }
            )

            # 2. Audio Guidance & Earphone Routing
            spoken_text, audio_path, audio_bytes = speak_vision_summary(
                vision_result=vision_result,
                engine=tts_engine,
                play_now=play_system
            )

            # 3. Update session state
            st.session_state["latest_analysis"] = vision_result
            st.session_state["annotated_img"] = draw_bounding_boxes(
                image,
                vision_result.get("detected_objects", [])
            )
            st.session_state["spoken_text"] = spoken_text
            st.session_state["audio_bytes"] = audio_bytes
            st.session_state["new_audio_ready"] = True
            st.session_state["latency"] = latency_sec
            st.session_state["entry_id"] = audit_entry["entry_id"]
            return True

        except Exception as ex:
            st.error(f"Analysis failed: {str(ex)}")
            return False


def render_sidebar():
    """Renders configuration settings and controls in the sidebar."""
    with st.sidebar:
        st.title("⚙️ Agent Settings")
        st.caption("Assistive Agent for Visually Impaired Navigation")

        # API Key Configuration
        env_key = os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "Gemini API Key",
            value=env_key,
            type="password",
            help="Optional: only needed if selecting the Cloud API (Gemini) backend or spoken Gemini audio parsing."
        )

        st.markdown("---")
        st.subheader("🤖 Vision Engine & Model")
        backend_choice = st.radio(
            "Inference Backend",
            options=[
                "Hybrid Local (RT-DETR + Florence-2)",
                "Local SLM Only (Florence-2)",
                "Cloud API (Gemini)",
            ],
            index=0,
            help="Hybrid pairs fast calibrated RT-DETR detection with Florence-2 scene narration locally."
        )
        if "Hybrid" in backend_choice:
            backend = "hybrid"
        elif "SLM" in backend_choice:
            backend = "slm"
        else:
            backend = "gemini"

        conf_threshold = 0.55
        if backend == "hybrid":
            selected_model = st.selectbox(
                "Local Detector Model (with Florence-2 Captioner)",
                options=[DEFAULT_DETECTOR_MODEL, "PekingU/rtdetr_v2_r50vd"],
                index=0,
                help="Calibrated local COCO object detector paired with Florence-2-base-ft captioning."
            )
            conf_threshold = st.slider(
                "Detector Confidence Threshold",
                min_value=0.25,
                max_value=0.85,
                value=0.55,
                step=0.05,
                help="Higher values (e.g. 0.55) suppress uncertain COCO guesses on non-COCO objects."
            )
        elif backend == "slm":
            selected_model = st.selectbox(
                "Local SLM Model",
                options=[DEFAULT_SLM_MODEL, "microsoft/Florence-2-base"],
                index=0,
                help="Pre-trained & fine-tuned Microsoft Florence-2 vision-language SLM."
            )
        else:
            selected_model = st.selectbox(
                "Gemini Model",
                options=[
                    "gemini-3.5-flash",
                    "gemini-3.5-flash-lite",
                    "gemini-3.8-flash",
                    "gemini-3-flash-preview",
                    "gemini-flash-latest",
                    "gemini-flash-lite-latest"
                ],
                index=0,
                help="Select the Gemini Vision model. 'gemini-3.5-flash' provides the fastest response."
            )

        st.markdown("---")
        st.subheader("🔊 Audio & Earphone Routing")
        tts_engine = st.selectbox(
            "Text-to-Speech Engine",
            options=["auto", "gtts", "pyttsx3"],
            index=0,
            help="'auto' selects gTTS with pyttsx3 fallback. Routes to Bluetooth earphones."
        )

        play_system = st.checkbox(
            "Play via System (Bluetooth Earphones)",
            value=True,
            help="Plays audio through the device's default audio output."
        )

        st.markdown("---")
        st.subheader("🎙️ Voice Trigger Sensitivity")
        enable_voice = st.checkbox(
            "Enable Hands-Free Voice Listener",
            value=True,
            help="Enables continuous listening for 'Capture', 'Describe', and 'What is in front of me?'."
        )

        st.markdown("---")
        st.subheader("📋 Audit Trail")
        history = load_audit_trail()
        st.metric("Logged Interactions", len(history))

        if st.button("🗑️ Clear Audit Trail", use_container_width=True):
            if clear_audit_trail():
                st.success("Audit trail cleared.")
                st.rerun()

    return api_key, backend, selected_model, conf_threshold, tts_engine, play_system, enable_voice


def main():
    api_key, backend, selected_model, conf_threshold, tts_engine, play_system, enable_voice = render_sidebar()
    analysis_ran_this_pass = False

    st.title("👁️ Assistive AI Agent with Voice Triggers")
    st.markdown(
        "Hands-free auditory vision guidance for the visually impaired. "
        "Say **'Capture'**, **'Describe'**, or **'Capture and describe'** to trigger navigation assistance."
    )

    # 1. Hands-Free In-Browser Voice Trigger Controller (Inlined cleanly without external asset loading)
    if enable_voice:
        components.html(get_voice_controller_html(), height=140)

    # 2. Voice Command Quick-Action Pills
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        voice_btn_cap_desc = st.button("🗣️ Voice: \"Capture & Describe\"", use_container_width=True, help="Simulate saying 'Capture and Describe'")
    with col_v2:
        voice_btn_desc = st.button("🗣️ Voice: \"Describe Scene\"", use_container_width=True, help="Simulate saying 'Describe'")
    with col_v3:
        voice_btn_repeat = st.button("🔁 Voice: \"Repeat Guidance\"", use_container_width=True, help="Simulate saying 'Repeat'")

    # Input Method Selection Tabs
    input_tab1, input_tab2, input_tab3 = st.tabs([
        "📷 Live Device Camera",
        "📁 Upload Image",
        "🎙️ Spoken Audio Command"
    ])

    selected_image: Optional[Image.Image] = st.session_state.get("current_input_image")
    image_name = st.session_state.get("current_input_image_name", "captured_scene.jpg")

    with input_tab1:
        st.caption("Point your camera forward. Say **'Capture'** or click below.")
        camera_file = st.camera_input("Device Camera Viewport")
        if camera_file is not None:
            selected_image = Image.open(camera_file).convert("RGB")
            image_name = f"camera_{int(time.time())}.jpg"
            st.session_state["current_input_image"] = selected_image
            st.session_state["current_input_image_name"] = image_name

    with input_tab2:
        uploaded_file = st.file_uploader(
            "Choose a scene image...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image to analyze the immediate surroundings."
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name
            st.session_state["current_input_image"] = selected_image
            st.session_state["current_input_image_name"] = image_name

    with input_tab3:
        st.write("### 🎙️ Spoken Voice Command (Gemini Audio Input)")
        st.markdown(
            "Speak your request into your microphone. Say phrases like:  \n"
            "• *'Capture and describe'*  \n"
            "• *'What is in front of me?'*  \n"
            "• *'Analyze surroundings'*  \n"
            "• *'Repeat guidance'*"
        )
        audio_clip = st.audio_input("Record Voice Command")
        if audio_clip is not None:
            with st.spinner("Processing speech command with Gemini..."):
                cmd_intent, transcript = parse_audio_command_with_gemini(audio_clip, api_key=api_key)
                st.success(f"🗣️ Transcribed Command: **\"{transcript}\"** (Intent: `{cmd_intent.value}`)")

                if cmd_intent in [VoiceCommand.CAPTURE_AND_DESCRIBE, VoiceCommand.DESCRIBE]:
                    target = selected_image or st.session_state.get("current_input_image")
                    if target and not analysis_ran_this_pass:
                        analysis_ran_this_pass = True
                        process_analysis_pipeline(
                            image=target,
                            image_name=image_name,
                            api_key=api_key,
                            selected_model=selected_model,
                            tts_engine=tts_engine,
                            play_system=play_system,
                            backend=backend,
                            confidence_threshold=conf_threshold,
                        )
                    elif not target:
                        st.warning("No image loaded to describe. Take a photo in the Camera tab first.")
                elif cmd_intent == VoiceCommand.REPEAT:
                    if "spoken_text" in st.session_state:
                        if play_system:
                            text_to_speech(st.session_state["spoken_text"], engine=tts_engine, play_now=True)
                        else:
                            st.session_state["new_audio_ready"] = True

    # Handle voice action triggers from buttons or URL query parameters
    query_cmd = st.query_params.get("voice_cmd")
    if query_cmd:
        st.query_params.clear()

    if voice_btn_cap_desc or query_cmd == "capture_and_describe":
        target = selected_image or st.session_state.get("current_input_image")
        if target and not analysis_ran_this_pass:
            analysis_ran_this_pass = True
            st.toast("🎙️ Executing Voice Command: 'Capture and Describe'")
            process_analysis_pipeline(
                image=target,
                image_name=image_name,
                api_key=api_key,
                selected_model=selected_model,
                tts_engine=tts_engine,
                play_system=play_system,
                backend=backend,
                confidence_threshold=conf_threshold,
            )
        elif not target:
            st.warning("No image loaded. Please take a photo or upload an image first.")

    elif voice_btn_desc or query_cmd == "describe":
        target = selected_image or st.session_state.get("current_input_image")
        if target and not analysis_ran_this_pass:
            analysis_ran_this_pass = True
            st.toast("🎙️ Executing Voice Command: 'Describe'")
            process_analysis_pipeline(
                image=target,
                image_name=image_name,
                api_key=api_key,
                selected_model=selected_model,
                tts_engine=tts_engine,
                play_system=play_system,
                backend=backend,
                confidence_threshold=conf_threshold,
            )
        elif not target:
            st.warning("No image loaded. Please take a photo or upload an image first.")

    elif voice_btn_repeat or query_cmd == "repeat":
        if "spoken_text" in st.session_state:
            st.toast("🔁 Replaying latest guidance narration")
            if play_system:
                text_to_speech(st.session_state["spoken_text"], engine=tts_engine, play_now=True)
            else:
                st.session_state["new_audio_ready"] = True

    # Active Image Display & Manual Trigger
    if selected_image is not None:
        col_img, col_btn = st.columns([2, 1])
        with col_img:
            st.image(selected_image, caption=f"Active Input: {image_name}", use_column_width=True)

        with col_btn:
            st.write("### Scene Ready")
            st.info(f"Resolution: {selected_image.width} × {selected_image.height} px")
            st.markdown("🗣️ *Say **'Describe'** or click below:*")
            trigger_button = st.button("🔍 Analyze Scene & Speak Guidance", type="primary", use_container_width=True)

        if trigger_button and not analysis_ran_this_pass:
            analysis_ran_this_pass = True
            process_analysis_pipeline(
                image=selected_image,
                image_name=image_name,
                api_key=api_key,
                selected_model=selected_model,
                tts_engine=tts_engine,
                play_system=play_system,
                backend=backend,
                confidence_threshold=conf_threshold,
            )

    # Display Results if available in session state
    if "latest_analysis" in st.session_state:
        st.markdown("---")
        analysis = st.session_state["latest_analysis"]
        spoken_text = st.session_state.get("spoken_text", "")
        audio_bytes = st.session_state.get("audio_bytes")
        latency = st.session_state.get("latency", 0.0)
        entry_id = st.session_state.get("entry_id", "")

        st.subheader("🔊 Auditory Guidance for User")
        st.markdown(f'<div class="summary-box">💬 <b>Spoken Output:</b><br>{spoken_text}</div>', unsafe_allow_html=True)

        if audio_bytes:
            # Avoid double-voice overlap: only autoplay in browser when system playback is off,
            # and consume new_audio_ready so subsequent UI reruns don't replay stale audio.
            new_audio = st.session_state.pop("new_audio_ready", False)
            audio_fmt = "audio/wav" if audio_bytes[:4] == b"RIFF" else "audio/mp3"
            st.audio(audio_bytes, format=audio_fmt, autoplay=(new_audio and not play_system))
            col_replay, col_meta = st.columns([1, 3])
            with col_replay:
                if st.button("🔁 Replay Guidance", key="replay_btn"):
                    if play_system:
                        text_to_speech(spoken_text, engine=tts_engine, play_now=True)
                    else:
                        st.session_state["new_audio_ready"] = True
                        st.rerun()
            with col_meta:
                st.caption(f"⚡ Processing Latency: **{latency}s** | 🏷️ Audit Entry ID: `{entry_id[:8]}...`")

        # Hazards & Spatial Safety
        reasoning = analysis.get("reasoning_and_confidence", {})
        hazards = reasoning.get("primary_hazards", [])

        if hazards and len(hazards) > 0:
            hazards_list = "<br>• ".join(hazards)
            st.markdown(f'<div class="hazard-box">⚠️ <b>Immediate Hazards Detected:</b><br>• {hazards_list}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">✅ <b>Path Status:</b> No immediate critical hazards detected. Proceed safely.</div>', unsafe_allow_html=True)

        # Tabular and Visual Breakdown
        tab_visual, tab_objects, tab_reasoning, tab_json = st.tabs([
            "🎯 Visual Overlay",
            "📦 Detected Objects",
            "🧠 Spatial Reasoning & Confidence",
            "📄 Raw JSON Output"
        ])

        with tab_visual:
            if "annotated_img" in st.session_state:
                st.image(
                    st.session_state["annotated_img"],
                    caption="Object Bounding Boxes (Normalized 0-1000 converted to pixel coordinates)",
                    use_column_width=True
                )

        with tab_objects:
            objects = analysis.get("detected_objects", [])
            if objects:
                obj_data = [
                    {
                        "Object": o.get("label", "").title(),
                        "Confidence": f"{int(o.get('confidence', 0.0) * 100)}%",
                        "Spatial Description": o.get("description", ""),
                        "Bounding Box": str(o.get("box_2d", []))
                    }
                    for o in objects
                ]
                st.table(obj_data)
            else:
                st.write("No distinct objects detected.")

        with tab_reasoning:
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Overall Confidence", f"{int(reasoning.get('overall_confidence', 0.0) * 100)}%")
                st.caption(f"Scene Type: **{reasoning.get('scene_type', 'N/A')}**")
            with col_r2:
                metrics = reasoning.get("confidence_metrics", {})
                st.metric("Detection Certainty", str(metrics.get("detection_certainty", "N/A")).title())
                st.caption(f"Lighting: **{reasoning.get('lighting_condition', 'N/A')}**")
            with col_r3:
                st.metric("Spatial Accuracy", str(metrics.get("spatial_accuracy", "N/A")).title())
                st.caption(f"Clarity: **{metrics.get('lighting_clarity', 'N/A')}**")

            st.write("**Reasoning Notes:**")
            st.info(reasoning.get("reasoning_notes", "No detailed reasoning provided."))

        with tab_json:
            st.json(analysis)

    # Audit Trail History Viewer
    with st.expander("📜 View Audit Trail Log (audit_log.json)"):
        records = load_audit_trail()
        if records:
            st.write(f"Displaying **{len(records)}** audit entries:")
            for item in reversed(records[-5:]):
                st.markdown(f"**Entry ID:** `{item.get('entry_id')}` | **Time:** `{item.get('timestamp_utc')}` | **Image:** `{item.get('image_identifier')}`")
                st.caption(f"Summary: *{item.get('model_output', {}).get('summary', 'N/A')}*")
                st.divider()

            audit_file = Path("audit_log.json")
            if audit_file.exists():
                with open(audit_file, "r", encoding="utf-8") as af:
                    st.download_button(
                        label="📥 Download Complete Audit Trail (JSON)",
                        data=af.read(),
                        file_name="audit_log.json",
                        mime="application/json"
                    )
        else:
            st.write("Audit log is currently empty.")


if __name__ == "__main__":
    main()
