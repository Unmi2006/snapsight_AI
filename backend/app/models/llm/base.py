"""
Shared types for the LLM layer. Mirrors app/models/ocr/base.py's pattern:
backend-agnostic result objects so llm_service.py can swap backends
without the API layer caring.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    max_new_tokens: int = 400
    temperature: float = 0.4
    top_p: float = 0.9


@dataclass
class LLMResult:
    text: str
    engine: str            # "onnx_genai"
    model_name: str        # display name of whichever exported folder loaded
    active_provider: str   # EP the loaded model folder declares -- see the
                            # honesty note in onnx_genai_backend.py before
                            # treating this the same way as OCR's verified EP
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
