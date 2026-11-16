"""
Audit Trail Module for Assistive AI Agent.

Maintains a secure, structured, and timestamped local audit log (audit_log.json)
for every visual analysis performed by the agent, ensuring full traceability
and compliance for healthcare/assistive technology prototypes.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure module logger
logger = logging.getLogger("AuditTrail")
DEFAULT_LOG_PATH = Path("audit_log.json")


def _read_existing_log(log_path: Path) -> List[Dict[str, Any]]:
    """
    Safely reads existing audit entries from the specified JSON file.
    Initializes a new empty list if the file is missing or unreadable.
    """
    if not log_path.exists():
        return []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # If wrapped in a dictionary container, extract or adapt
                return [data]
            else:
                logger.warning("Unrecognized audit log format, resetting to empty list.")
                return []
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error reading existing audit log from {log_path}: {e}")
        # Create a backup of the corrupted file before returning empty list
        backup_path = log_path.with_suffix(f".corrupt_{int(datetime.now().timestamp())}.bak")
        try:
            log_path.rename(backup_path)
            logger.info(f"Backed up corrupted log to {backup_path}")
        except Exception:
            pass
        return []


def log_interaction(
    image_identifier: str,
    gemini_output: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    log_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Appends an interaction record to the local audit trail JSON file.

    Parameters:
        image_identifier (str): Filename, hash, or camera capture identifier.
        gemini_output (dict): The exact structured JSON response received from Gemini 1.5 Flash.
        metadata (dict, optional): Additional contextual metadata (e.g., latency, audio status, user ID).
        log_file_path (str, optional): Custom file path for the audit log (defaults to 'audit_log.json').

    Returns:
        dict: The complete audit entry that was committed to the log.
    """
    path = Path(log_file_path) if log_file_path else DEFAULT_LOG_PATH

    entry_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    audit_entry: Dict[str, Any] = {
        "entry_id": entry_id,
        "timestamp_utc": timestamp_utc,
        "image_identifier": str(image_identifier),
        "model_output": gemini_output,
        "metadata": metadata or {}
    }

    try:
        # Load existing entries
        records = _read_existing_log(path)
        records.append(audit_entry)

        # Write atomically using a temporary file in the same directory
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        temp_path.replace(path)
        logger.info(f"Interaction successfully logged. Entry ID: {entry_id} -> {path}")
    except Exception as e:
        logger.error(f"Failed to append to audit log {path}: {e}")
        # In-memory return still succeeds so pipeline flow isn't interrupted
        audit_entry["log_error"] = str(e)

    return audit_entry


def load_audit_trail(log_file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all interaction records stored in the audit trail.

    Parameters:
        log_file_path (str, optional): Custom file path for the audit log.

    Returns:
        list: List of historical audit entry dictionaries (chronological order).
    """
    path = Path(log_file_path) if log_file_path else DEFAULT_LOG_PATH
    return _read_existing_log(path)


def clear_audit_trail(log_file_path: Optional[str] = None) -> bool:
    """
    Clears all entries in the audit trail file (useful for fresh benchmark or test runs).

    Parameters:
        log_file_path (str, optional): Custom file path for the audit log.

    Returns:
        bool: True if cleared successfully, False otherwise.
    """
    path = Path(log_file_path) if log_file_path else DEFAULT_LOG_PATH
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        logger.info(f"Audit log cleared: {path}")
        return True
    except OSError as e:
        logger.error(f"Failed to clear audit log {path}: {e}")
        return False


if __name__ == "__main__":
    # Self-test when executed directly
    print("[AuditTrail] Running self-test...")
    sample_response = {
        "summary": "Clear hallway with an open doorway at 12 o'clock.",
        "detected_objects": [
            {"label": "door", "confidence": 0.96, "box_2d": [100, 200, 800, 500]}
        ],
        "reasoning_and_confidence": {
            "overall_confidence": 0.95,
            "primary_hazards": []
        }
    }
    result = log_interaction("test_image_001.jpg", sample_response, {"test_run": True})
    print(f"Logged entry ID: {result['entry_id']}")
    history = load_audit_trail()
    print(f"Total audit entries: {len(history)}")
