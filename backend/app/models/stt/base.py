"""
Shared types for every STT backend (PocketSphinx now, Whisper ONNX once
exported). Mirrors app/models/ocr/base.py's pattern so stt_service.py can
swap implementations without the API layer caring.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class STTResult:
    text: str
    engine: str                       # "pocketsphinx" or "whisper_onnx"
    active_provider: str              # honest EP that actually ran
    latency_ms: float
    language: str = "en"
