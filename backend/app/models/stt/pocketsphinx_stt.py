"""
Classical CPU speech recognition via CMU PocketSphinx (through the
SpeechRecognition package). Deliberately NOT an NPU path -- exactly like
TesseractOCR in Phase 1, this exists so voice INPUT works end-to-end
today with zero model downloads: PocketSphinx's default English
acoustic/language model ships INSIDE the pocketsphinx pip wheel itself,
no separate export or download step needed.

Accuracy is noticeably lower than the Whisper ONNX path
(whisper_onnx.py) -- this is purely the always-available fallback while
that path is being set up, never claimed to be anything more.
"""
from __future__ import annotations

import time

import numpy as np

from app.models.stt.base import STTResult


class PocketSphinxSTT:
    name = "pocketsphinx"

    def __init__(self):
        import speech_recognition as sr

        self._sr = sr
        self._recognizer = sr.Recognizer()

        # Fail fast and clearly if pocketsphinx itself (the actual decoder,
        # not just the SpeechRecognition wrapper) isn't importable, rather
        # than a cryptic error on the first real request.
        import pocketsphinx  # noqa: F401

    @staticmethod
    def is_available() -> bool:
        try:
            import pocketsphinx  # noqa: F401
            import speech_recognition  # noqa: F401

            return True
        except ImportError:
            return False

    def run(self, samples: np.ndarray, sample_rate: int = 16000) -> STTResult:
        t0 = time.perf_counter()
        int16 = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)
        audio_data = self._sr.AudioData(int16.tobytes(), sample_rate, 2)

        try:
            text = self._recognizer.recognize_sphinx(audio_data)
        except self._sr.UnknownValueError:
            text = ""  # honest empty result, not a fake placeholder string
        latency_ms = (time.perf_counter() - t0) * 1000

        return STTResult(
            text=text,
            engine=self.name,
            active_provider="CPUExecutionProvider",  # honest: PocketSphinx never uses NPU/GPU
            latency_ms=round(latency_ms, 3),
        )
