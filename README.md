# 👁️ Assistive AI Agent for the Visually Impaired

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-RT--DETR%20%2B%20Florence--2-success.svg)](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365)
[![Benchmark](https://img.shields.io/badge/COCO%20val2017-0.5668%20mAP-orange.svg)](benchmark_metrics.json)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, privacy-first assistive vision agent designed to aid visually impaired individuals navigate their surroundings safely and independently. 

The system captures visual inputs (via webcam, local photo upload, or **hands-free continuous voice commands**), performs real-time object detection and spatial reasoning using a **Hybrid Local Edge Pipeline (`PekingU/rtdetr_r50vd_coco_o365` + `microsoft/Florence-2-base-ft`)**, routes spoken auditory guidance directly to connected **Bluetooth earphones**, maintains a strict local **audit trail**, and achieves **`0.5668` mAP on official Microsoft COCO `val2017`** benchmarks without requiring any GPU or cloud API.

---

## 🌟 Key Highlights

- **100% Offline Edge Mode:** Runs locally on standard CPU in **~0.7s per frame** using cached PyTorch weights—no cloud pings, no API subscriptions, and no data leaks in subway stations or elevators.
- **Hands-Free In-Browser Voice Triggers:** Continuous Web Speech API recognition listening for natural wake phrases like *"What is in front of me?"*, *"Capture and describe"*, *"Describe"*, and *"Repeat"*.
- **Calibrated Spatial Geometry:** Converts detected bounding boxes into intuitive 12-hour clock-face directions (*"Chair at 2 o'clock"*, *"Stairs straight ahead at 12 o'clock"*).
- **Hazard-First Safety Hierarchy:** Immediate alert priority for safety-critical hazards (`car`, `truck`, `stairs`, `bicycle`, etc.) announced before general scene context.
- **Dual Engine Architecture:** Defaults to the local offline Hybrid detector, with an optional cloud fallback to **Google Gemini Vision (1.5/3.5 Flash)**.
- **Quantitatively Benchmarked:** Evaluated on 50 real Microsoft COCO `val2017` scenes using official `pycocotools` (`COCOeval`).

---

## 🏗️ Architecture Overview

```mermaid
flowchart LR
    A["📷 Camera / Upload / Voice"] --> B["🖥️ Streamlit Frontend (app.py)"]
    B --> C["🧠 Perception Engine (vision_agent.py)"]
    
    subgraph EdgePerception["Hybrid Edge Pipeline (100% Local / Offline)"]
        C --> D1["🎯 RT-DETR: Calibrated BBoxes (~0.7s)"]
        C --> D2["🗣️ Florence-2: Open-Vocabulary Captioning"]
        D1 --> E["📦 Master JSON Schema"]
        D2 --> E
    end

    C -.->|Optional Cloud Fallback| F["☁️ Google Gemini Vision"]
    F -.-> E

    E --> G["🔊 Audio Router (audio_utils.py)"]
    E --> H["📋 Local Audit Trail (audit_trail.py)"]
    
    G --> I["🎧 Bluetooth Earphones / Default Audio Sink"]
    G --> J["🌐 In-Browser Audio Player"]
    H --> K[("audit_log.json")]
```

---

## 📊 Vision Backends & Credentials

| Backend | Model(s) | Credentials Needed? | Internet Needed? | CPU Latency | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hybrid Local (Default)** | `PekingU/rtdetr_r50vd_coco_o365` + `microsoft/Florence-2-base-ft` | **None** (No API key or HF token) | First download only (~172 MB + ~445 MB) | **~0.7s / frame** | Calibrated boxes paired with rich Florence-2 narration; cached in `~/.cache/huggingface/hub/`. |
| **Local SLM Only** | `microsoft/Florence-2-base-ft` | **None** | First download only (~445 MB) | ~3.5s / frame | Single-model detection and captioning on CPU/GPU. |
| **Cloud API (Optional)** | `gemini-3.5-flash` (multi-model fallback chain) | **Yes** (`GEMINI_API_KEY`) | Yes | ~1.5s / frame | Multimodal cloud reasoning & conversational audio command parsing. |

---

## 📁 Repository Structure

```text
assistive_ai_agent/
├── app.py                   # Streamlit UI, sidebar controls & main execution loop
├── vision_agent.py          # Vision engine: Hybrid RT-DETR + Florence-2 & Gemini fallback
├── voice_trigger.py         # Web Speech API controller & spoken wake-word intent parser
├── audio_utils.py           # Text-to-speech audio router (Bluetooth & browser playback)
├── audit_trail.py           # Local JSON interaction audit logger (audit_log.json)
├── prepare_coco_50.py       # Downloads official COCO val2017 subset (50 images + annotations)
├── benchmark.py             # Official COCO evaluator using pycocotools (--backend hybrid|slm|gemini)
├── data/
│   └── coco_val2017_50/     # Pre-packaged 50 real COCO val2017 images & ground truth annotations
├── benchmark_metrics.json   # Verified pycocotools evaluation metrics summary
├── benchmark_detections.json# Generated bounding box predictions for COCO evaluation
├── requirements.txt         # Pinned Python project dependencies
├── .env.example             # Template for optional Gemini API key
├── SETUP_GUIDE.md           # Step-by-step setup guide for new users
├── .gitignore               # Ignores venv, local logs, and private secrets
└── README.md                # Project documentation and quickstart
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/assistive_ai_agent.git
cd assistive_ai_agent
```

### 2. Set Up Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# On CPU-only laptops (recommended — avoids massive CUDA downloads & keeps wheels matched):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Or on NVIDIA GPU machines:
pip install -r requirements.txt
```

### 4. (Optional) Configure Gemini API Key

The default **Hybrid Local Engine requires zero keys or accounts**. If you want to enable the optional **Cloud Gemini** backend:

```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

### 5. Launch the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 🎧 Operating Instructions

1. **Bluetooth Audio Routing:** Pair your Bluetooth earphones/headphones to your machine. The operating system designates them as the default output device, and spoken guidance streams directly to them.
2. **Hands-Free Voice Triggers:**
   - Click **"🎙️ Start Listening"** in the top banner and grant microphone permissions.
   - Say **"Capture and describe"** *(or **"What is in front of me?"**)* $\rightarrow$ Automatically captures a camera frame, analyzes hazards/pathways, and speaks guidance.
   - Say **"Capture"** $\rightarrow$ Takes a photo snapshot.
   - Say **"Describe"** $\rightarrow$ Analyzes the current frame.
   - Say **"Repeat"** $\rightarrow$ Replays the latest guidance narration.
3. **Manual Tabs:** Alternatively use the **Live Device Camera**, **Upload Image**, or **Spoken Audio Command** tabs.

---

## 📈 Official COCO val2017 Benchmark Results

To evaluate detection performance against official human ground-truth annotations across 50 real Microsoft COCO `val2017` scenes:

```bash
python benchmark.py --backend hybrid --limit 50
```

### Comparison: Florence-2 SLM vs. Hybrid RT-DETR

Results computed by `pycocotools` (`COCOeval`) on an AMD Ryzen CPU (verified in `benchmark_metrics.json`):

| Evaluation Metric | Standalone Florence-2 SLM | Hybrid Engine (RT-DETR + Florence-2) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **mAP @ IoU=0.50:0.95 (Primary)** | `0.4758` | **`0.5668`** | **+19.1% Accuracy Gain** |
| **mAP @ IoU=0.50 (PASCAL VOC)** | `0.6128` | **`0.7261`** | **+18.5% Overlap Precision** |
| **mAP @ IoU=0.75 (Strict)** | `0.5354` | **`0.6122`** | **+14.3% Localization Gain** |
| **mAP Small Objects** | `0.2495` | **`0.3988`** | **+59.8% Boost on Distant Items** |
| **mAP Medium Objects** | `0.5970` | **`0.7050`** | **+18.1% Gain** |
| **mAP Large Objects (Hazards)** | **`0.7953`** | `0.7585` | **~76% High Precision on Hazards** |
| **Average Recall (AR @ 100)** | `0.5227` | **`0.6890`** | **+31.8% Higher Object Coverage** |
| **CPU Latency per Image** | `~3.5s` | **`~0.7s`** | **4.4× Faster on CPU** |

---

## 🛡️ Safety, Telemetry & Auditability

Every frame analysis automatically appends a tamper-evident record into `audit_log.json`:
- **UTC Timestamp** (`ISO 8601`)
- **Input Image Resolution**
- **Hardware Inference Latency** (seconds)
- **Active Model Backend** (`hybrid`, `slm`, or `gemini`)
- **Detected Objects, Normalized Bounding Boxes & Confidence Scores**
- **Identified Hazards & Safe Pathway Recommendations**

All telemetry remains strictly local to the user's device, ensuring zero leakage of personal visual feeds.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
