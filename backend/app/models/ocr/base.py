"""
Shared types for every OCR backend (Tesseract now, PaddleOCR-ONNX once
exported). Keeping this backend-agnostic is what lets ocr_service.py swap
implementations without the API layer caring.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextBox:
    text: str
    confidence: float          # 0-100
    left: int
    top: int
    width: int
    height: int


@dataclass
class OCRResult:
    full_text: str
    boxes: list[TextBox] = field(default_factory=list)
    engine: str = "unknown"          # "tesseract" or "paddle_ocr_onnx"
    active_provider: str = "CPUExecutionProvider"  # honest EP that actually ran
    latency_ms: float = 0.0
