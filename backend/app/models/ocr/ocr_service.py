"""
Single entry point the API layer calls for OCR. Decides, once at startup,
whether the ONNX/NPU-capable recognizer is available; if not, transparently
uses Tesseract so the feature still works. The choice is logged and
reported back in every response -- never silently pretend one ran when it
didn't.
"""
from __future__ import annotations

import logging

from PIL import Image

from app.core.inference_engine import engine as inference_engine
from app.benchmarking.profiler import profile_call
from app.models.ocr.base import OCRResult
from app.models.ocr.tesseract_ocr import TesseractOCR

logger = logging.getLogger("snapsight.ocr_service")


class OCRService:
    def __init__(self):
        self._onnx_backend = None
        self._tesseract_backend: TesseractOCR | None = None
        self._active_backend_name = None
        self._unavailable_reason: str | None = None
        self._init_backends()

    def _init_backends(self):
        from app.models.ocr.paddle_ocr_onnx import PaddleOCRONNX

        if PaddleOCRONNX.is_available():
            try:
                self._onnx_backend = PaddleOCRONNX(inference_engine)
                self._active_backend_name = "paddle_ocr_onnx"
                logger.info("OCR: using ONNX recognizer via InferenceEngine.")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR: ONNX backend failed to init (%s); falling back to Tesseract.", exc)

        # Don't let a missing Tesseract binary take the whole backend down
        # (it used to). Camera/document/vision endpoints check
        # `is_available` and return a clean 503 instead of the server
        # failing to start.
        try:
            self._tesseract_backend = TesseractOCR()
            self._active_backend_name = "tesseract"
            logger.info("OCR: rec.onnx not found -- using Tesseract (CPU) until it's exported.")
        except Exception as exc:  # noqa: BLE001
            self._unavailable_reason = str(exc)
            logger.warning("OCR: no backend available -- %s", exc)

    @property
    def active_backend_name(self) -> str:
        return self._active_backend_name

    @property
    def is_available(self) -> bool:
        return self._onnx_backend is not None or self._tesseract_backend is not None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    def _current_provider(self) -> str:
        if self._onnx_backend:
            return self._onnx_backend.handle.load_result.active_provider
        return "CPUExecutionProvider"

    def run(self, image: Image.Image) -> OCRResult:
        backend = self._onnx_backend or self._tesseract_backend
        if backend is None:
            raise RuntimeError(self._unavailable_reason or "No OCR backend available.")
        with profile_call(model_name=f"ocr:{backend.name}", active_provider=self._current_provider()):
            result = backend.run(image)
        return result


# Module-level singleton, created once when the app starts.
ocr_service = OCRService()
