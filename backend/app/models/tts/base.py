"""
Shared types for every TTS backend (pyttsx3 now, Piper ONNX once
exported). Mirrors app/models/stt/base.py's pattern so tts_service.py can
swap implementations without the API layer caring.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTSResult:
    audio_bytes: bytes                # a complete, standalone WAV file
    sample_rate: int
    engine: str                       # "pyttsx3" or "piper_onnx"
    active_provider: str              # honest EP that actually ran
    latency_ms: float
