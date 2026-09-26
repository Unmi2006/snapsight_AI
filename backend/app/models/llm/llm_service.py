"""
Single entry point the API layer calls for text generation. Mirrors
app/models/ocr/ocr_service.py's selection pattern (try backends in
priority order, log what happened, report the truth) with one deliberate
difference: OCR has Tesseract as an always-available zero-download
fallback, so Phase 1 could let a missing backend raise during startup.
There is no equivalent always-available, non-fake local LLM fallback --
so LLMService NEVER raises out of __init__. If no model is installed, it
just marks itself unavailable and the /api/vision/study endpoint reports
that honestly instead of returning a canned or hardcoded "AI" answer.
"""
from __future__ import annotations

import logging

from app.benchmarking.profiler import profile_call
from app.core.config import LLM_EP_PRIORITY, LLM_MODELS_DIR
from app.models.llm.base import GenerationConfig, LLMResult

logger = logging.getLogger("snapsight.llm_service")


class LLMUnavailableError(RuntimeError):
    """Raised by generate() when no local LLM backend could be loaded."""


class LLMService:
    def __init__(self, preference: str = "auto"):
        self._backend = None
        self._unavailable_reason: str | None = None
        self._preference = preference
        self._init_backend()

    def _init_backend(self):
        from app.models.llm.onnx_genai_backend import OnnxGenAILLM

        # Phase 4 (Settings): "auto" walks LLM_EP_PRIORITY exactly as
        # before. Anything else FORCES that one subdir -- if it isn't
        # installed, this reports unavailable naming the forced choice
        # rather than silently trying the others, so an explicit
        # provider override in Settings actually means what it says.
        if self._preference == "auto":
            candidates = LLM_EP_PRIORITY
        else:
            candidates = [(s, ep) for s, ep in LLM_EP_PRIORITY if s == self._preference]
            if not candidates:
                self._unavailable_reason = (
                    f"Unknown LLM provider preference '{self._preference}'."
                )
                logger.warning("LLM: %s", self._unavailable_reason)
                return

        attempts: list[str] = []
        for subdir, ep_label in candidates:
            model_dir = LLM_MODELS_DIR / subdir
            if not OnnxGenAILLM.is_available(model_dir):
                attempts.append(f"{subdir} ({ep_label}): not found, or onnxruntime-genai not installed")
                continue
            try:
                self._backend = OnnxGenAILLM(model_dir, model_display_name=f"local-llm-{subdir}")
                logger.info("LLM: using '%s' backend from %s", subdir, model_dir)
                return
            except Exception as exc:  # noqa: BLE001 -- try the next EP folder
                attempts.append(f"{subdir} ({ep_label}): found but failed to load -- {exc}")
                logger.warning("LLM backend '%s' failed to load: %s", subdir, exc)

        forced_note = (
            "" if self._preference == "auto"
            else f" (Settings is forcing provider='{self._preference}' -- switch it back to 'auto' to allow the others.)"
        )
        self._unavailable_reason = (
            "No local LLM model is loaded yet. Checked: " + "; ".join(attempts) + "." + forced_note + " "
            "Run `python scripts/export_llm_model.py` on a machine with internet "
            "access to fetch/export a model, then restart the backend. Study Vision's "
            "OCR-only features still work without this -- only the 'ask AI' step needs it."
        )
        logger.warning("LLM: %s", self._unavailable_reason)

    def reload(self, preference: str = "auto") -> None:
        """Phase 4 (Settings): re-run backend selection under a new
        provider preference. Called by app/api/settings.py after a
        settings update -- never called automatically."""
        self._backend = None
        self._unavailable_reason = None
        self._preference = preference
        self._init_backend()

    @staticmethod
    def available_backends() -> list[dict]:
        """Which LLM_EP_PRIORITY subdirs actually have an exported model
        on disk right now -- drives the Settings page's dropdown so it
        never offers a provider that would just fail to load."""
        from app.models.llm.onnx_genai_backend import OnnxGenAILLM

        return [
            {
                "id": subdir,
                "label": ep_label,
                "installed": OnnxGenAILLM.is_available(LLM_MODELS_DIR / subdir),
            }
            for subdir, ep_label in LLM_EP_PRIORITY
        ]

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    @property
    def preference(self) -> str:
        return self._preference

    def status(self) -> dict:
        if self._backend:
            return {
                "available": True,
                "model_name": self._backend.model_display_name,
                "active_provider": self._backend.declared_provider,
                "load_time_ms": self._backend.load_time_ms,
                "preference": self._preference,
            }
        return {"available": False, "reason": self._unavailable_reason, "preference": self._preference}

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> LLMResult:
        if not self._backend:
            raise LLMUnavailableError(self._unavailable_reason)
        with profile_call(
            model_name=f"llm:{self._backend.name}",
            active_provider=self._backend.declared_provider,
        ):
            result = self._backend.generate(prompt, config)
        return result


# Module-level singleton, created once when the app starts, honoring
# whatever provider preference was saved in Settings (Phase 4) on the
# previous run. Safe even with no model installed -- see the docstring
# above. Import is local to avoid a circular import at module load time
# (app.storage.settings_store has no dependency back on this module, but
# keeping the import here mirrors how the other services stay decoupled).
from app.storage.settings_store import get_settings as _get_settings  # noqa: E402

llm_service = LLMService(preference=_get_settings().get("llm_provider_preference", "auto"))
