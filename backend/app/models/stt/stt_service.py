"""
Single entry point the API layer calls for speech-to-text. Same
selection pattern as OCRService (Phase 1): try the NPU/GPU-capable ONNX
path first, transparently fall back to the always-available CPU
classical engine if it isn't exported yet, and never crash the whole
backend if NEITHER is available -- report unavailable and let
/api/voice/* return a clean 503 instead of the server failing to start.
"""
from __future__ import annotations

import logging

import numpy as np

from app.benchmarking.profiler import profile_call
from app.core.config import STT_MODEL_DIR
from app.core.inference_engine import engine as inference_engine
from app.models.stt.base import STTResult

logger = logging.getLogger("snapsight.stt_service")


class STTService:
    def __init__(self, preference: str = "auto"):
        self._onnx_backend = None
        self._sphinx_backend = None
        self._active_backend_name = None
        self._unavailable_reason: str | None = None
        self._preference = preference
        self._init_backends()

    def _init_backends(self):
        from app.models.stt.whisper_onnx import OnnxWhisperSTT

        pref = self._preference
        if pref not in ("auto", "whisper_onnx", "pocketsphinx"):
            self._unavailable_reason = f"Unknown STT backend preference '{pref}'."
            logger.warning("STT: %s", self._unavailable_reason)
            return

        # Phase 4 (Settings): "auto" keeps the original whisper-then-sphinx
        # fallback chain. Forcing one name skips the other entirely --
        # honestly reporting unavailable if the forced one isn't installed,
        # rather than quietly using whichever else happens to work.
        try_whisper = pref in ("auto", "whisper_onnx")
        try_sphinx = pref in ("auto", "pocketsphinx")

        if try_whisper and OnnxWhisperSTT.is_available(STT_MODEL_DIR):
            try:
                self._onnx_backend = OnnxWhisperSTT(STT_MODEL_DIR, inference_engine)
                self._active_backend_name = "whisper_onnx"
                logger.info("STT: using Whisper ONNX via InferenceEngine.")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("STT: ONNX Whisper backend failed to init (%s); falling back to PocketSphinx.", exc)
                if pref == "whisper_onnx":
                    self._unavailable_reason = f"Whisper ONNX was forced in Settings but failed to load: {exc}"
                    return

        if try_sphinx:
            try:
                from app.models.stt.pocketsphinx_stt import PocketSphinxSTT

                self._sphinx_backend = PocketSphinxSTT()
                self._active_backend_name = "pocketsphinx"
                logger.info("STT: using PocketSphinx (CPU).")
                return
            except Exception as exc:  # noqa: BLE001
                self._unavailable_reason = (
                    f"No STT backend available ({exc}). Run "
                    "`python scripts/export_stt_model.py` for the NPU/GPU-capable "
                    "Whisper path, or `pip install SpeechRecognition pocketsphinx` "
                    "for the always-available CPU fallback."
                )
                logger.warning("STT: %s", self._unavailable_reason)
                return

        forced_note = "" if pref == "auto" else f" (Settings is forcing preference='{pref}'.)"
        self._unavailable_reason = (
            "No STT backend matched the current preference." + forced_note
        )
        logger.warning("STT: %s", self._unavailable_reason)

    def reload(self, preference: str = "auto") -> None:
        """Phase 4 (Settings): re-run backend selection under a new
        preference. Called by app/api/settings.py, never automatically."""
        self._onnx_backend = None
        self._sphinx_backend = None
        self._active_backend_name = None
        self._unavailable_reason = None
        self._preference = preference
        self._init_backends()

    @staticmethod
    def available_backends() -> list[dict]:
        """Which STT backends actually work right now -- drives the
        Settings page's dropdown honestly."""
        from app.models.stt.pocketsphinx_stt import PocketSphinxSTT
        from app.models.stt.whisper_onnx import OnnxWhisperSTT

        return [
            {"id": "whisper_onnx", "label": "Whisper (ONNX, NPU/GPU-capable)",
             "installed": OnnxWhisperSTT.is_available(STT_MODEL_DIR)},
            {"id": "pocketsphinx", "label": "PocketSphinx (CPU, always-on fallback)",
             "installed": PocketSphinxSTT.is_available()},
        ]

    @property
    def active_backend_name(self) -> str | None:
        return self._active_backend_name

    @property
    def is_available(self) -> bool:
        return self._onnx_backend is not None or self._sphinx_backend is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def preference(self) -> str:
        return self._preference

    def status(self) -> dict:
        if not self.is_available:
            return {"available": False, "reason": self._unavailable_reason, "preference": self._preference}
        backend = self._onnx_backend or self._sphinx_backend
        return {
            "available": True,
            "engine": self._active_backend_name,
            "active_provider": getattr(backend, "active_provider", "CPUExecutionProvider"),
            "preference": self._preference,
        }

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> STTResult:
        backend = self._onnx_backend or self._sphinx_backend
        if backend is None:
            raise RuntimeError(self._unavailable_reason or "No STT backend available.")
        provider = getattr(backend, "active_provider", "CPUExecutionProvider")
        with profile_call(model_name=f"stt:{backend.name}", active_provider=provider):
            result = backend.run(samples, sample_rate)
        return result


# Module-level singleton, created once when the app starts, honoring
# whatever backend preference was saved in Settings (Phase 4) on the
# previous run. Safe even with no backend available -- see the docstring
# above.
from app.storage.settings_store import get_settings as _get_settings  # noqa: E402

stt_service = STTService(preference=_get_settings().get("stt_backend_preference", "auto"))
