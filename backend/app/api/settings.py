"""
/api/settings -- Phase 4: Settings page (model selection, provider
preference overrides, voice selection).

This is intentionally a single GET/PUT/POST-reset trio rather than one
endpoint per field: the Settings page always shows the whole form at
once, and every field change needs the same "persist, then reload the
affected service" round trip. `available` in the GET response is built
live from each service's `available_backends()` (never a hardcoded list)
so the Settings page can never offer a provider that isn't actually
installed on this machine -- same no-fake-claims rule as everywhere
else in this backend.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.llm.llm_service import llm_service
from app.models.stt.stt_service import stt_service
from app.models.tts.tts_service import tts_service
from app.orchestrator.study_vision import list_modes
from app.storage import settings_store

router = APIRouter(prefix="/api/settings", tags=["settings"])

_VALID_LLM_PREFS = {"auto", "qnn", "dml", "cpu"}
_VALID_STT_PREFS = {"auto", "whisper_onnx", "pocketsphinx"}
_VALID_TTS_PREFS = {"auto", "piper_onnx", "pyttsx3"}


class SettingsPatch(BaseModel):
    """All fields optional -- PUT only applies whichever ones are sent,
    leaving the rest untouched (a partial update, not a full replace)."""

    llm_provider_preference: Optional[str] = None
    stt_backend_preference: Optional[str] = None
    tts_backend_preference: Optional[str] = None
    tts_voice_id: Optional[str] = Field(None, description="Pass \"\" to clear back to the engine default voice.")
    default_study_vision_mode: Optional[str] = None
    default_max_new_tokens: Optional[int] = Field(None, ge=16, le=1024)
    auto_speak_voice_responses: Optional[bool] = None


def _build_response() -> dict:
    return {
        "settings": settings_store.get_settings(),
        "available": {
            "llm_providers": llm_service.available_backends(),
            "stt_backends": stt_service.available_backends(),
            "tts_backends": tts_service.available_backends(),
            "tts_voices": tts_service.available_voices(),
            "study_vision_modes": list_modes(),
        },
        "live_status": {
            "llm": llm_service.status(),
            "stt": stt_service.status(),
            "tts": tts_service.status(),
        },
    }


@router.get("")
def get_settings():
    return _build_response()


@router.put("")
def update_settings(patch: SettingsPatch):
    updates = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No settings fields were provided to update.")

    if "llm_provider_preference" in updates and updates["llm_provider_preference"] not in _VALID_LLM_PREFS:
        raise HTTPException(400, f"llm_provider_preference must be one of {sorted(_VALID_LLM_PREFS)}.")
    if "stt_backend_preference" in updates and updates["stt_backend_preference"] not in _VALID_STT_PREFS:
        raise HTTPException(400, f"stt_backend_preference must be one of {sorted(_VALID_STT_PREFS)}.")
    if "tts_backend_preference" in updates and updates["tts_backend_preference"] not in _VALID_TTS_PREFS:
        raise HTTPException(400, f"tts_backend_preference must be one of {sorted(_VALID_TTS_PREFS)}.")
    if "default_study_vision_mode" in updates:
        valid_modes = {m["id"] for m in list_modes()}
        if updates["default_study_vision_mode"] not in valid_modes:
            raise HTTPException(400, f"default_study_vision_mode must be one of {sorted(valid_modes)}.")
    if updates.get("tts_voice_id") == "":
        updates["tts_voice_id"] = None

    new_settings = settings_store.update_settings(updates)

    # Only reload the services whose relevant setting actually changed --
    # each reload rebuilds a model/session, so this avoids an unnecessary
    # reload (e.g. of the LLM) when only the TTS voice changed.
    if "llm_provider_preference" in updates:
        llm_service.reload(preference=new_settings["llm_provider_preference"])
    if "stt_backend_preference" in updates:
        stt_service.reload(preference=new_settings["stt_backend_preference"])
    if "tts_backend_preference" in updates or "tts_voice_id" in updates:
        tts_service.reload(
            preference=new_settings["tts_backend_preference"],
            voice_id=new_settings["tts_voice_id"],
        )

    return _build_response()


@router.post("/reset")
def reset_settings():
    defaults = settings_store.reset_settings()
    llm_service.reload(preference=defaults["llm_provider_preference"])
    stt_service.reload(preference=defaults["stt_backend_preference"])
    tts_service.reload(preference=defaults["tts_backend_preference"], voice_id=defaults["tts_voice_id"])
    return _build_response()
