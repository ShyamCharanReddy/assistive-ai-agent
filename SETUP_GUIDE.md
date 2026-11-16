# Quick Setup & Execution Guide for New Users

Welcome! You have received the **Assistive AI Agent for the Visually Impaired**. By default, the vision engine runs **locally** on your machine using a **Hybrid pipeline**: **`PekingU/rtdetr_r50vd_coco_o365`** for fast calibrated object detection + **`microsoft/Florence-2-base-ft`** for natural scene descriptions—no Hugging Face account, token, or paid API key is required.

---

## Prerequisites

- **Python 3.10, 3.11, or 3.12** installed on your system (`python --version` or `python3 --version`).
- At least **~2–3 GB free disk space** (for PyTorch/Transformers + the ~445 MB Florence-2 model cache).
- A webcam/camera and microphone (built-in or external).
- *(Optional)* Connected Bluetooth earphones or headphones for private audio guidance.
- *(Optional)* A **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/) only if you want to switch to the Cloud Gemini backend.

---

## Step-by-Step Installation

### Step 1: Open Terminal / Command Prompt
Navigate to the project folder:

```bash
cd path/to/assistive_ai_agent
```

### Step 2: Create & Activate a Virtual Environment

**On Windows (PowerShell or Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

*(You will see `(venv)` appear at the beginning of your terminal prompt).*

### Step 3: Install Required Dependencies

If your computer **does not have an NVIDIA GPU** (CPU-only laptop/desktop), install CPU PyTorch wheels first so you don't download multi-GB CUDA libraries and to keep `torch`/`torchvision` versions matched:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If you **do have an NVIDIA GPU**, you can install directly:

```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Configure Gemini API Key
Skip this step if you only plan to use the **Local SLM (Florence-2)** backend.

If you want to also use **Cloud API (Gemini)**:
1. Copy `.env.example` to `.env`:
   - **Windows**: `copy .env.example .env`
   - **macOS / Linux**: `cp .env.example .env`
2. Open `.env` and paste your key:
   ```env
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   ```

### Step 5: Launch the Application

```bash
streamlit run app.py
```

Your browser will open `http://localhost:8501`.

> **First-Run Download Note:** On the very first Hybrid Local run, `transformers` automatically downloads the public `PekingU/rtdetr_r50vd_coco_o365` (~172 MB) and `microsoft/Florence-2-base-ft` (~445 MB) weights into your local Hugging Face cache (`~/.cache/huggingface/hub/`). **No Hugging Face login or token is needed**, and every run after that works **100% offline**.

---

## How to Use the Agent

1. **Check the Sidebar Mode**:
   - Under **Vision Engine & Model**, **Inference Backend** defaults to **Hybrid Local (RT-DETR + Florence-2)** (runs offline locally using cached weights).
2. **Connect Bluetooth Earphones**:
   - Pair your earphones/headphones so they act as your system's default audio output.
3. **Hands-Free Voice Triggers**:
   - Click **"🎙️ Start Listening"** in the top voice banner and allow microphone permissions.
   - Say **"Capture and describe"** *(or **"What is in front of me?"**)* → Captures a frame, detects objects/hazards, and speaks guidance aloud.
   - Say **"Capture"** → Snaps a photo.
   - Say **"Describe"** → Analyzes the current photo.
   - Say **"Repeat"** → Replays the latest speech.
4. **Alternative Input Tabs**:
   - **Live Device Camera**, **Upload Image**, or **Spoken Audio Command** (uses Gemini if configured).

---

## Running the 50-Image Real COCO Benchmark (Optional)

1. Download 50 real COCO `val2017` images and official annotations (~250 MB temporary download during prep; zip is auto-cleaned):
   ```bash
   python prepare_coco_50.py --output-dir data/coco_val2017_50 --limit 50
   ```
2. Run `pycocotools` evaluation using the default Hybrid RT-DETR backend (no API key needed):
   ```bash
   python benchmark.py --backend hybrid --limit 50
   ```
   *(On CPU-only laptops, expect ~40 seconds total for 50 images, saving results to `benchmark_metrics.json`).*

---

## Troubleshooting

- **`torchvision::nms does not exist` error**: Reinstall matching CPU `torch` and `torchvision` wheels:
  ```bash
  pip uninstall -y torchvision
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- **Offline cache wait**: When running completely offline without internet, `transformers` may take ~15 seconds on startup before falling back to the local cache (`local_files_only=True`).
- **Microphone / Camera Permission**: Click **"Allow"** when your browser prompts for camera/microphone access.
- **Port Busy**: Launch on another port if `8501` is taken:
  `streamlit run app.py --server.port 8502`
