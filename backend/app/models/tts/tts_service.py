"""
Single entry point the API layer calls for text-to-speech. Same
selection pattern as OCRService/STTService: try the NPU/GPU-capable ONNX
path (Piper) first, transparently fall back to the always-available CPU
classical engine (pyttsx3) if it isn't exported yet, and never crash the
whole backend if NEITHER is available -- report unavailable and let
/api/voice/* return a clean 503 instead of the server failing to start.
"""
from __future__ import annotations

import logging

from app.benchmarking.profiler import profile_call
from app.core.config import TTS_MODEL_DIR
from app.core.inference_engine import engine as inference_engine
from app.models.tts.base import TTSResult

logger = logging.getLogger("snapsight.tts_service")


class TTSService:
    def __init__(self, preference: str = "auto", voice_id: str | None = None):
        self._onnx_backend = None
        self._pyttsx3_backend = None
        self._active_backend_name = None
        self._unavailable_reason: str | None = None
        self._preference = preference
        self._voice_id = voice_id
        self._init_backends()

    def _init_backends(self):
        from app.models.tts.piper_onnx_tts import PiperOnnxTTS

        pref = self._preference
        if pref not in ("auto", "piper_onnx", "pyttsx3"):
            self._unavailable_reason = f"Unknown TTS backend preference '{pref}'."
            logger.warning("TTS: %s", self._unavailable_reason)
            return

        # Phase 4 (Settings): same forced-vs-auto semantics as STT/LLM
        # above -- "auto" keeps the original Piper-then-pyttsx3 chain,
        # forcing one name skips the other and reports honestly if the
        # forced choice isn't installed.
        try_piper = pref in ("auto", "piper_onnx")
        try_pyttsx3 = pref in ("auto", "pyttsx3")

        if try_piper and PiperOnnxTTS.is_available(TTS_MODEL_DIR):
            try:
                self._onnx_backend = PiperOnnxTTS(TTS_MODEL_DIR, inference_engine)
                self._active_backend_name = "piper_onnx"
                logger.info("TTS: using Piper ONNX via InferenceEngine.")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("TTS: Piper ONNX backend failed to init (%s); falling back to pyttsx3.", exc)
                if pref == "piper_onnx":
                    self._unavailable_reason = f"Piper ONNX was forced in Settings but failed to load: {exc}"
                    return

        if try_pyttsx3:
            try:
                from app.models.tts.pyttsx3_tts import Pyttsx3TTS

                self._pyttsx3_backend = Pyttsx3TTS(voice_id=self._voice_id)
                self._active_backend_name = "pyttsx3"
                logger.info("TTS: using pyttsx3 (CPU)%s.", f" with voice '{self._voice_id}'" if self._voice_id else "")
                return
            except Exception as exc:  # noqa: BLE001
                self._unavailable_reason = (
                    f"No TTS backend available ({exc}). Run "
                    "`python scripts/export_tts_model.py` for the NPU/GPU-capable "
                    "Piper path, or `pip install pyttsx3` (plus a native speech "
                    "engine like espeak-ng on Linux) for the always-available CPU fallback."
                )
                logger.warning("TTS: %s", self._unavailable_reason)
                return

        forced_note = "" if pref == "auto" else f" (Settings is forcing preference='{pref}'.)"
        self._unavailable_reason = "No TTS backend matched the current preference." + forced_note
        logger.warning("TTS: %s", self._unavailable_reason)

    def reload(self, preference: str = "auto", voice_id: str | None = None) -> None:
        """Phase 4 (Settings): re-run backend selection under a new
        preference/voice. Called by app/api/settings.py, never automatically."""
        self._onnx_backend = None
        self._pyttsx3_backend = None
        self._active_backend_name = None
        self._unavailable_reason = None
        self._preference = preference
        self._voice_id = voice_id
        self._init_backends()

    @staticmethod
    def available_backends() -> list[dict]:
        """Which TTS backends actually work right now -- drives the
        Settings page's dropdown honestly."""
        from app.models.tts.piper_onnx_tts import PiperOnnxTTS
        from app.models.tts.pyttsx3_tts import Pyttsx3TTS

        return [
            {"id": "piper_onnx", "label": "Piper (ONNX, NPU/GPU-capable)",
             "installed": PiperOnnxTTS.is_available(TTS_MODEL_DIR)},
            {"id": "pyttsx3", "label": "pyttsx3 (CPU, native OS voice)",
             "installed": Pyttsx3TTS.is_available()},
        ]

    @staticmethod
    def available_voices() -> list[dict]:
        """pyttsx3 voice options for the current OS, for the Settings
        page's voice picker. Empty list if pyttsx3 itself isn't usable."""
        from app.models.tts.pyttsx3_tts import Pyttsx3TTS

        return Pyttsx3TTS.list_voices()

    @property
    def active_backend_name(self) -> str | None:
        return self._active_backend_name

    @property
    def is_available(self) -> bool:
        return self._onnx_backend is not None or self._pyttsx3_backend is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def preference(self) -> str:
        return self._preference

    def status(self) -> dict:
        if not self.is_available:
            return {"available": False, "reason": self._unavailable_reason, "preference": self._preference}
        backend = self._onnx_backend or self._pyttsx3_backend
        return {
            "available": True,
            "engine": self._active_backend_name,
            "active_provider": getattr(backend, "active_provider", "CPUExecutionProvider"),
            "preference": self._preference,
        }

    def synthesize(self, text: str) -> TTSResult:
        backend = self._onnx_backend or self._pyttsx3_backend
        if backend is None:
            raise RuntimeError(self._unavailable_reason or "No TTS backend available.")
        provider = getattr(backend, "active_provider", "CPUExecutionProvider")
        with profile_call(model_name=f"tts:{backend.name}", active_provider=provider):
            result = backend.run(text)
        return result


# Module-level singleton, created once when the app starts, honoring
# whatever backend/voice preference was saved in Settings (Phase 4) on
# the previous run. Safe even with no backend available -- see the
# docstring above.
from app.storage.settings_store import get_settings as _get_settings  # noqa: E402

_settings = _get_settings()
tts_service = TTSService(
    preference=_settings.get("tts_backend_preference", "auto"),
    voice_id=_settings.get("tts_voice_id"),
)
