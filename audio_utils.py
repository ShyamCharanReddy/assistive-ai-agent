"""
Audio Router Module for Assistive AI Agent.

Parses descriptive visual summaries, converts them into high-fidelity speech,
and automatically routes the audio stream through the operating system's default
audio sink (connected Bluetooth earphones or speakers) as well as providing
audio bytes for web-based playback.
"""

import io
import logging
import os
import platform
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("AudioRouter")

# Directory for storing synthesized speech files
AUDIO_CACHE_DIR = Path(tempfile.gettempdir()) / "assistive_ai_audio"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _play_system_audio_file(file_path: str) -> bool:
    """
    Plays an audio file through the operating system's default audio sink,
    which automatically routes sound to connected Bluetooth earphones or default speakers.
    """
    system_os = platform.system().lower()

    # Strategy 1: Try pygame mixer (cross-platform, non-blocking / smooth)
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        # Wait for audio playback to finish in background thread or wait loop
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        return True
    except Exception as e:
        logger.debug(f"Pygame playback unavailable or failed: {e}. Trying system CLI fallbacks.")

    # Strategy 2: Native OS CLI audio players
    try:
        if "linux" in system_os:
            # Check popular Linux CLI audio players
            for player in ["mpg123", "paplay", "mpv", "ffplay", "aplay"]:
                if subprocess.run(["which", player], capture_output=True).returncode == 0:
                    if player == "ffplay":
                        cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path]
                    elif player == "mpv":
                        cmd = ["mpv", "--no-terminal", file_path]
                    else:
                        cmd = [player, file_path]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
        elif "darwin" in system_os:  # macOS
            subprocess.run(["afplay", file_path], check=True)
            return True
        elif "windows" in system_os:  # Windows
            import winsound
            if file_path.endswith(".wav"):
                winsound.PlaySound(file_path, winsound.SND_FILENAME)
                return True
            else:
                # Use Windows Media Player or PowerShell command
                cmd = ["powershell", "-c", f'(New-Object Media.SoundPlayer "{file_path}").PlaySync()']
                subprocess.run(cmd, check=True)
                return True
    except Exception as err:
        logger.warning(f"Native CLI audio playback encountered an issue: {err}")

    return False


def _synthesize_with_gtts(text: str, output_path: str) -> bool:
    """
    Synthesizes speech using Google Text-to-Speech (gTTS).
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        logger.error(f"gTTS synthesis failed: {e}")
        return False


def _synthesize_and_play_pyttsx3(text: str, output_path: Optional[str] = None) -> bool:
    """
    Synthesizes and speaks text directly using pyttsx3 (offline system speech synthesizer).
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)     # Natural conversational pace for assistive tech
        engine.setProperty("volume", 1.0)   # Full volume for Bluetooth earphones

        if output_path:
            engine.save_to_file(text, output_path)
            engine.runAndWait()
        else:
            engine.say(text)
            engine.runAndWait()
        return True
    except Exception as e:
        logger.error(f"pyttsx3 execution failed: {e}")
        return False


def text_to_speech(
    text: str,
    engine: str = "auto",
    play_now: bool = True
) -> Tuple[Optional[str], Optional[bytes]]:
    """
    Converts text to spoken audio, saves an audio file, and plays it through system output.

    Parameters:
        text (str): Spoken narration text.
        engine (str): TTS engine ('auto', 'gtts', or 'pyttsx3'). Default: 'auto'.
        play_now (bool): If True, immediately routes sound to system output / Bluetooth earphones.

    Returns:
        tuple (file_path: Optional[str], audio_bytes: Optional[bytes]):
            Path to saved audio file and raw bytes for frontend playback.
    """
    if not text or not text.strip():
        logger.warning("Empty text passed to text_to_speech. Skipping.")
        return None, None

    clean_text = text.strip()
    cache_file = AUDIO_CACHE_DIR / f"speech_{hash(clean_text) & 0xffffffff}.mp3"
    audio_path = str(cache_file)
    audio_bytes: Optional[bytes] = None

    # Determine synthesis strategy
    succeeded = False
    chosen_engine = engine.lower()

    if chosen_engine in ["auto", "gtts"]:
        # Primary: gTTS for natural inflection
        succeeded = _synthesize_with_gtts(clean_text, audio_path)
        if not succeeded and chosen_engine == "auto":
            # Fallback to pyttsx3 if offline or gTTS fails
            logger.info("Falling back to pyttsx3 offline synthesizer.")
            wav_path = str(cache_file.with_suffix(".wav"))
            succeeded = _synthesize_and_play_pyttsx3(clean_text, wav_path)
            if succeeded:
                audio_path = wav_path
    elif chosen_engine == "pyttsx3":
        wav_path = str(cache_file.with_suffix(".wav"))
        succeeded = _synthesize_and_play_pyttsx3(clean_text, wav_path)
        if succeeded:
            audio_path = wav_path
        else:
            # Fallback to gTTS
            succeeded = _synthesize_with_gtts(clean_text, audio_path)

    # Read bytes for web playback
    if Path(audio_path).exists():
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            logger.warning(f"Could not read audio file bytes: {e}")

    # Route immediately to system audio (Bluetooth earphones) if requested
    if play_now and Path(audio_path).exists():
        # Play in a background thread so UI doesn't freeze
        play_thread = threading.Thread(
            target=_play_system_audio_file,
            args=(audio_path,),
            daemon=True
        )
        play_thread.start()

    return audio_path, audio_bytes


def speak_vision_summary(
    vision_result: Dict[str, Any],
    engine: str = "auto",
    play_now: bool = True
) -> Tuple[str, Optional[str], Optional[bytes]]:
    """
    Parses the descriptive summary and urgent hazards from the vision agent JSON output,
    constructs an optimized auditory guidance message, and speaks it.

    Parameters:
        vision_result (dict): Structured JSON response from vision_agent.py.
        engine (str): 'auto', 'gtts', or 'pyttsx3'.
        play_now (bool): Whether to play immediately through system audio output.

    Returns:
        tuple (spoken_script: str, audio_file_path: Optional[str], audio_bytes: Optional[bytes])
    """
    summary = vision_result.get("summary", "").strip()

    # Check for immediate primary hazards to prioritize safety announcements
    reasoning = vision_result.get("reasoning_and_confidence", {})
    hazards = reasoning.get("primary_hazards", [])

    spoken_parts = []
    if hazards and len(hazards) > 0:
        hazard_text = ", ".join(hazards)
        spoken_parts.append(f"Caution: Detected {hazard_text}.")

    if summary:
        spoken_parts.append(summary)
    else:
        spoken_parts.append("Scene analyzed. No detailed description available.")

    full_narration = " ".join(spoken_parts)
    logger.info(f"Generating voice guidance: {full_narration[:80]}...")

    audio_path, audio_bytes = text_to_speech(
        text=full_narration,
        engine=engine,
        play_now=play_now
    )

    return full_narration, audio_path, audio_bytes


if __name__ == "__main__":
    print("[AudioRouter] Running audio test...")
    sample_agent_output = {
        "summary": "Walkway ahead is clear for 3 meters. A desk is on your right.",
        "reasoning_and_confidence": {
            "primary_hazards": ["small electrical cord on floor"]
        }
    }
    script, path, raw_bytes = speak_vision_summary(sample_agent_output, play_now=False)
    print(f"Generated narration script:\n'{script}'")
    print(f"Audio file saved at: {path}")
    print(f"Byte stream size: {len(raw_bytes) if raw_bytes else 0} bytes")
    print("Self-test completed successfully.")
