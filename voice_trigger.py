"""
Voice Trigger & Speech Command Module for Assistive AI Agent.

Enables hands-free voice-activated capture and scene description for visually impaired users.
Features fuzzy intent matching for wake words ('capture', 'describe', 'analyze', etc.),
Gemini multimodal audio command parsing, spoken audio acknowledgments, and an inline
Web Speech API voice controller.
"""

import enum
import io
import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("VoiceTrigger")


class VoiceCommand(enum.Enum):
    """Recognized voice command intents for assistive interaction."""
    NONE = "none"
    CAPTURE = "capture"
    DESCRIBE = "describe"
    CAPTURE_AND_DESCRIBE = "capture_and_describe"
    REPEAT = "repeat"
    CLEAR = "clear"


# Keyword regex patterns for instantaneous local intent matching
PATTERNS = {
    VoiceCommand.CAPTURE_AND_DESCRIBE: [
        r"capture and describe",
        r"take (?:a )?photo and (?:describe|analyze)",
        r"snap and describe",
        r"what is in front of me",
        r"what's in front of me",
        r"what is ahead",
        r"what's ahead",
        r"what do you see",
        r"look and describe",
        r"scan scene",
        r"help me see"
    ],
    VoiceCommand.CAPTURE: [
        r"^capture$",
        r"take (?:a )?(?:photo|picture|shot)",
        r"^snap$",
        r"^shoot$",
        r"capture (?:image|photo|frame)",
        r"take photo now"
    ],
    VoiceCommand.DESCRIBE: [
        r"^describe$",
        r"^analyze$",
        r"describe (?:scene|image|surroundings)",
        r"analyze (?:scene|image|surroundings)",
        r"what is this",
        r"read scene",
        r"explain scene"
    ],
    VoiceCommand.REPEAT: [
        r"^repeat$",
        r"^replay$",
        r"say again",
        r"repeat guidance",
        r"repeat description",
        r"read again"
    ],
    VoiceCommand.CLEAR: [
        r"^clear$",
        r"clear log",
        r"reset",
        r"start over"
    ]
}


def parse_voice_transcript(text: str) -> Tuple[VoiceCommand, str]:
    """
    Parses a recognized voice transcript into an actionable VoiceCommand intent.
    """
    if not text:
        return VoiceCommand.NONE, ""

    clean_text = text.lower().strip()
    clean_text = re.sub(r"^(?:hey agent|assistant|gemini|please|agent)[,\s]+", "", clean_text)

    # 1. Check composite 'capture and describe' first
    for pattern in PATTERNS[VoiceCommand.CAPTURE_AND_DESCRIBE]:
        if re.search(pattern, clean_text):
            logger.info(f"Matched CAPTURE_AND_DESCRIBE intent: '{clean_text}'")
            return VoiceCommand.CAPTURE_AND_DESCRIBE, clean_text

    # 2. Check standalone capture
    for pattern in PATTERNS[VoiceCommand.CAPTURE]:
        if re.search(pattern, clean_text):
            logger.info(f"Matched CAPTURE intent: '{clean_text}'")
            return VoiceCommand.CAPTURE, clean_text

    # 3. Check standalone describe
    for pattern in PATTERNS[VoiceCommand.DESCRIBE]:
        if re.search(pattern, clean_text):
            logger.info(f"Matched DESCRIBE intent: '{clean_text}'")
            return VoiceCommand.DESCRIBE, clean_text

    # 4. Check repeat
    for pattern in PATTERNS[VoiceCommand.REPEAT]:
        if re.search(pattern, clean_text):
            logger.info(f"Matched REPEAT intent: '{clean_text}'")
            return VoiceCommand.REPEAT, clean_text

    # 5. Check clear
    for pattern in PATTERNS[VoiceCommand.CLEAR]:
        if re.search(pattern, clean_text):
            logger.info(f"Matched CLEAR intent: '{clean_text}'")
            return VoiceCommand.CLEAR, clean_text

    return VoiceCommand.NONE, clean_text


def parse_audio_command_with_gemini(
    audio_data: Any,
    api_key: Optional[str] = None
) -> Tuple[VoiceCommand, str]:
    """
    Transcribes audio and extracts the intent using Gemini multimodal capabilities.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        return VoiceCommand.NONE, "API key missing"

    try:
        import google.generativeai as genai
    except ImportError:
        return VoiceCommand.NONE, "google-generativeai not installed"

    if hasattr(audio_data, "read"):
        audio_data.seek(0)
        audio_bytes = audio_data.read()
    elif isinstance(audio_data, bytes):
        audio_bytes = audio_data
    else:
        return VoiceCommand.NONE, "Unsupported audio data format"

    if not audio_bytes:
        return VoiceCommand.NONE, "Empty audio buffer"

    genai.configure(api_key=resolved_key, transport="rest")

    model = genai.GenerativeModel(
        model_name="gemini-3.5-flash",
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    )

    prompt = """
    Listen to this user audio recording.
    Determine what the user said (transcript) and classify their intent into one of these strict categories:
    - "CAPTURE_AND_DESCRIBE": User wants to capture a photo and describe/analyze what is in front of them (e.g. "what's in front of me", "capture and describe", "look ahead", "help me see").
    - "CAPTURE": User wants to take a photo (e.g. "capture", "take photo", "snap").
    - "DESCRIBE": User wants to analyze or describe the current scene (e.g. "describe", "analyze", "explain").
    - "REPEAT": User wants to replay or repeat the audio description.
    - "OTHER": None of the above.

    Return JSON matching:
    {
      "transcript": "Exact words spoken by user",
      "intent": "CAPTURE_AND_DESCRIBE / CAPTURE / DESCRIBE / REPEAT / OTHER"
    }
    """

    audio_part = {
        "mime_type": "audio/wav",
        "data": audio_bytes
    }

    try:
        response = model.generate_content([prompt, audio_part])
        text_content = response.text.strip()
        data = json.loads(text_content)
        transcript = data.get("transcript", "")
        intent_str = str(data.get("intent", "OTHER")).upper()

        if "CAPTURE_AND_DESCRIBE" in intent_str:
            return VoiceCommand.CAPTURE_AND_DESCRIBE, transcript
        elif "CAPTURE" in intent_str:
            return VoiceCommand.CAPTURE, transcript
        elif "DESCRIBE" in intent_str:
            return VoiceCommand.DESCRIBE, transcript
        elif "REPEAT" in intent_str:
            return VoiceCommand.REPEAT, transcript
        else:
            cmd, _ = parse_voice_transcript(transcript)
            return cmd, transcript

    except Exception as e:
        logger.warning(f"Gemini audio intent parsing failed: {e}")
        return VoiceCommand.NONE, f"Error: {e}"


def get_command_acknowledgment(command: VoiceCommand) -> str:
    """Returns spoken acknowledgment message to confirm receipt of command."""
    if command == VoiceCommand.CAPTURE_AND_DESCRIBE:
        return "Capturing scene and analyzing surroundings."
    elif command == VoiceCommand.CAPTURE:
        return "Taking photo now."
    elif command == VoiceCommand.DESCRIBE:
        return "Analyzing current scene."
    elif command == VoiceCommand.REPEAT:
        return "Repeating guidance."
    elif command == VoiceCommand.CLEAR:
        return "Resetting session."
    return ""


def get_voice_controller_html() -> str:
    """
    Returns an inline, self-contained HTML/JS widget for in-browser speech recognition.
    Directly embedded via streamlit.components.v1.html to avoid any custom component
    handshake timeouts or deployment asset loading issues.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <style>
    body {
      margin: 0;
      padding: 6px 12px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: transparent;
      color: #111;
    }
    .voice-panel {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      padding: 12px 18px;
      background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
      border: 2px solid #0d6efd;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(13, 110, 253, 0.08);
    }
    .status-group {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .pulse-dot {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background-color: #6c757d;
      display: inline-block;
      transition: background-color 0.3s;
    }
    .pulse-dot.listening {
      background-color: #28a745;
      box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7);
      animation: pulse 1.4s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
      70% { transform: scale(1.15); box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }
    .status-text {
      font-size: 14.5px;
      font-weight: 600;
      color: #0d47a1;
    }
    .last-heard {
      font-size: 13px;
      color: #495057;
      margin-top: 3px;
    }
    .toggle-btn {
      padding: 8px 18px;
      font-size: 14px;
      font-weight: 600;
      border-radius: 6px;
      border: none;
      cursor: pointer;
      background-color: #0d6efd;
      color: white;
      transition: all 0.2s ease;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .toggle-btn:hover {
      background-color: #0b5ed7;
    }
    .toggle-btn.active {
      background-color: #dc3545;
    }
    .commands-list {
      font-size: 12px;
      color: #555;
      margin-top: 4px;
      width: 100%;
    }
    kbd {
      background: #e9ecef;
      border: 1px solid #ced4da;
      border-radius: 3px;
      padding: 1px 5px;
      font-size: 11px;
      font-family: monospace;
      color: #212529;
    }
  </style>
</head>
<body>

<div class="voice-panel">
  <div class="status-group">
    <span id="dot" class="pulse-dot"></span>
    <div>
      <div id="status-label" class="status-text">Voice Trigger: Ready</div>
      <div id="heard-label" class="last-heard">Click to start hands-free listening.</div>
    </div>
  </div>

  <button id="listen-btn" class="toggle-btn" aria-label="Toggle Continuous Voice Trigger">
    🎙️ Start Listening
  </button>

  <div class="commands-list">
    🗣️ <b>Voice Wake Phrases:</b> 
    <kbd>Capture</kbd> &nbsp;|&nbsp; 
    <kbd>Describe</kbd> &nbsp;|&nbsp; 
    <kbd>Capture and Describe</kbd> &nbsp;|&nbsp; 
    <kbd>What's in front of me?</kbd> &nbsp;|&nbsp;
    <kbd>Repeat</kbd>
  </div>
</div>

<script>
  function playEarcon(type) {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.connect(gain);
      gain.connect(ctx.destination);

      if (type === 'success') {
        osc.frequency.setValueAtTime(587.33, ctx.currentTime);
        osc.frequency.setValueAtTime(880.00, ctx.currentTime + 0.12);
        gain.gain.setValueAtTime(0.2, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.35);
      } else {
        osc.frequency.setValueAtTime(440, ctx.currentTime);
        gain.gain.setValueAtTime(0.12, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.15);
      }
    } catch (e) {
      console.warn("AudioContext error:", e);
    }
  }

  let lastTriggerTs = 0;

  function triggerStreamlitAction(command) {
    const now = Date.now();
    if (now - lastTriggerTs < 4000) {
      console.log("Ignoring duplicate voice trigger within cooldown:", command);
      return;
    }
    lastTriggerTs = now;
    playEarcon('success');
    console.log("Triggering Streamlit action:", command);

    // Strategy 1: Find parent buttons and trigger native click cleanly (no double URL dispatch)
    let triggeredDom = false;
    try {
      const parentDoc = window.parent.document;
      const buttons = Array.from(parentDoc.querySelectorAll('button'));

      if (command === 'capture' || command === 'capture_and_describe') {
        const camBtn = parentDoc.querySelector('button[data-testid="stCameraInputButton"]')
                    || buttons.find(b => b.innerText && (b.innerText.includes('Take Photo') || b.innerText.includes('Take photo')));
        if (camBtn) {
          camBtn.click();
          triggeredDom = true;
          console.log("Clicked camera button");
        }
      }

      if (command === 'describe' || command === 'capture_and_describe') {
        const delay = (command === 'capture_and_describe') ? 700 : 100;
        const analyzeBtn = buttons.find(b => b.innerText && b.innerText.includes('Analyze Scene'))
                        || buttons.find(b => b.innerText && b.innerText.includes('Describe Scene'));
        if (analyzeBtn) {
          triggeredDom = true;
          setTimeout(() => {
            analyzeBtn.click();
            console.log("Clicked analyze button");
          }, delay);
        }
      }

      if (command === 'repeat') {
        const replayBtn = buttons.find(b => b.innerText && b.innerText.includes('Replay'));
        if (replayBtn) {
          replayBtn.click();
          triggeredDom = true;
        }
      }
    } catch (err) {
      console.warn("Parent DOM button click fallback:", err);
    }

    // Strategy 2: Only fall back to URL query params if DOM button could not be clicked
    if (!triggeredDom) {
      try {
        const url = new URL(window.parent.location.href);
        url.searchParams.set("voice_cmd", command);
        url.searchParams.set("v_ts", Date.now().toString());
        window.parent.history.replaceState({}, "", url.toString());
      } catch (e) {
        console.warn("URL params update fallback:", e);
      }
    }
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let isListening = false;

  const dot = document.getElementById('dot');
  const statusLabel = document.getElementById('status-label');
  const heardLabel = document.getElementById('heard-label');
  const listenBtn = document.getElementById('listen-btn');

  if (!SpeechRecognition) {
    statusLabel.innerText = "Speech API not supported in this browser";
    heardLabel.innerText = "Use the 'Spoken Audio Command' tab below for voice interaction.";
    listenBtn.disabled = true;
  } else {
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = function() {
      isListening = true;
      dot.className = 'pulse-dot listening';
      statusLabel.innerText = "Listening for 'Capture' or 'Describe'...";
      heardLabel.innerText = "Say: 'Capture and describe', 'Capture', or 'Describe'";
      listenBtn.innerText = "🛑 Stop Listening";
      listenBtn.className = "toggle-btn active";
      playEarcon('start');
    };

    recognition.onend = function() {
      if (isListening) {
        try { recognition.start(); } catch(e) {}
      } else {
        dot.className = 'pulse-dot';
        statusLabel.innerText = "Voice Trigger: Paused";
        listenBtn.innerText = "🎙️ Start Listening";
        listenBtn.className = "toggle-btn";
      }
    };

    recognition.onerror = function(event) {
      console.warn("Recognition error:", event.error);
      if (event.error === 'not-allowed') {
        isListening = false;
        dot.className = 'pulse-dot';
        statusLabel.innerText = "Microphone access blocked";
        heardLabel.innerText = "Please allow microphone permissions in your browser.";
        listenBtn.className = "toggle-btn";
      }
    };

    recognition.onresult = function(event) {
      const results = event.results;
      const lastResult = results[results.length - 1];
      if (!lastResult.isFinal) return;

      const raw = lastResult[0].transcript.trim().toLowerCase();
      heardLabel.innerHTML = `Heard: <b>"${raw}"</b>`;

      if (raw.includes("capture and describe") || 
          raw.includes("what is in front of me") || 
          raw.includes("what's in front of me") || 
          raw.includes("what do you see") ||
          raw.includes("what is ahead") ||
          raw.includes("what's ahead") ||
          raw.includes("help me see")) {
        triggerStreamlitAction("capture_and_describe");
      } else if (raw.includes("capture") || raw.includes("take photo") || raw.includes("snap")) {
        triggerStreamlitAction("capture");
      } else if (raw.includes("describe") || raw.includes("analyze") || raw.includes("explain")) {
        triggerStreamlitAction("describe");
      } else if (raw.includes("repeat") || raw.includes("replay") || raw.includes("say again")) {
        triggerStreamlitAction("repeat");
      }
    };

    listenBtn.addEventListener('click', function() {
      if (isListening) {
        isListening = false;
        try { recognition.stop(); } catch(e) {}
      } else {
        try { recognition.start(); } catch(e) {
          console.warn("Could not start speech recognition:", e);
        }
      }
    });
  }
</script>
</body>
</html>
    """


if __name__ == "__main__":
    print("[VoiceTrigger] Self-testing intent parsing...")
    assert parse_voice_transcript("Capture")[0] == VoiceCommand.CAPTURE
    assert parse_voice_transcript("Describe scene")[0] == VoiceCommand.DESCRIBE
    assert parse_voice_transcript("Capture and describe")[0] == VoiceCommand.CAPTURE_AND_DESCRIBE
    assert parse_voice_transcript("What is in front of me?")[0] == VoiceCommand.CAPTURE_AND_DESCRIBE
    print("All tests passed! HTML template length:", len(get_voice_controller_html()))
