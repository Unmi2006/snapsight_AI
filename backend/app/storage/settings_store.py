"""
Phase 4: Settings storage.

A single persisted JSON document (one row in the `settings` table),
not one row per key -- there are few enough settings that this is
simpler than a real key/value schema, and it means a read is always one
query. app/api/settings.py is the only place that validates and applies
these; this module just persists whatever dict it's given.
"""
from __future__ import annotations

import json

from app.storage.db import get_connection

_ROW_KEY = "app_settings"

DEFAULT_SETTINGS: dict = {
    # "auto" = InferenceEngine/LLMService's own NPU->GPU->CPU priority order
    # (app/core/config.py's *_PRIORITY lists). Anything else FORCES that one
    # backend and reports unavailable -- honestly -- rather than silently
    # falling back, so an override actually means what it says.
    "llm_provider_preference": "auto",   # "auto" | "qnn" | "dml" | "cpu"
    "stt_backend_preference": "auto",    # "auto" | "whisper_onnx" | "pocketsphinx"
    "tts_backend_preference": "auto",    # "auto" | "piper_onnx" | "pyttsx3"
    "tts_voice_id": None,                # pyttsx3 voice id, or None = engine default
    "default_study_vision_mode": "explain",
    "default_max_new_tokens": 400,
    "auto_speak_voice_responses": True,
}


def get_settings() -> dict:
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (_ROW_KEY,)).fetchone()
    conn.close()
    if row is None:
        return dict(DEFAULT_SETTINGS)
    try:
        stored = json.loads(row["value"])
    except json.JSONDecodeError:
        return dict(DEFAULT_SETTINGS)
    # Merge over defaults so a settings row saved by an older version of
    # this app (missing a key added later) still comes back complete.
    merged = dict(DEFAULT_SETTINGS)
    merged.update(stored)
    return merged


def save_settings(settings: dict) -> dict:
    conn = get_connection()
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_ROW_KEY, json.dumps(settings)),
    )
    conn.commit()
    conn.close()
    return settings


def update_settings(patch: dict) -> dict:
    current = get_settings()
    current.update(patch)
    return save_settings(current)


def reset_settings() -> dict:
    return save_settings(dict(DEFAULT_SETTINGS))
