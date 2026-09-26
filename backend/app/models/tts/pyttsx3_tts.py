"""
Classical CPU text-to-speech via pyttsx3, which wraps each OS's native
offline speech engine (SAPI5 on Windows, NSSpeechSynthesizer on macOS,
espeak/espeak-ng on Linux). Deliberately NOT an NPU path -- exactly like
TesseractOCR/PocketSphinxSTT elsewhere in this codebase, it exists so
voice OUTPUT works end-to-end today, zero model downloads, while the
ONNX/NPU-accelerated neural TTS path (piper_onnx_tts.py) is being set up.
"""
from __future__ import annotations

import os
import tempfile
import time
import wave

from app.models.tts.base import TTSResult


class Pyttsx3TTS:
    name = "pyttsx3"

    def __init__(self, voice_id: str | None = None):
        import pyttsx3

        self._pyttsx3 = pyttsx3
        self._voice_id = voice_id
        # Fail fast if no native speech engine is registered on this OS
        # (e.g. espeak-ng missing on a minimal Linux box) rather than a
        # confusing error on the first real request. Also validates the
        # requested voice_id (if any) actually exists on this machine --
        # Phase 4 (Settings) picks voice_id from list_voices(), but the
        # OS's voice set can change between runs, so we check rather than
        # assume.
        engine = pyttsx3.init()
        if voice_id:
            available_ids = {v.id for v in engine.getProperty("voices")}
            if voice_id not in available_ids:
                engine.stop()
                raise ValueError(f"Voice id '{voice_id}' is not installed on this machine.")
        engine.stop()

    @staticmethod
    def is_available() -> bool:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.stop()
            return True
        except Exception:  # noqa: BLE001 -- any native-engine init failure means unavailable
            return False

    @staticmethod
    def list_voices() -> list[dict]:
        """Voices the native OS engine actually has installed -- used to
        populate the Settings page's voice picker so it never offers a
        voice that doesn't exist on this machine."""
        try:
            import pyttsx3

            engine = pyttsx3.init()
            voices = [
                {"id": v.id, "name": v.name, "languages": getattr(v, "languages", [])}
                for v in engine.getProperty("voices")
            ]
            engine.stop()
            return voices
        except Exception:  # noqa: BLE001
            return []

    def run(self, text: str) -> TTSResult:
        t0 = time.perf_counter()
        engine = self._pyttsx3.init()
        if self._voice_id:
            engine.setProperty("voice", self._voice_id)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            engine.save_to_file(text, path)
            engine.runAndWait()
            with open(path, "rb") as f:
                audio_bytes = f.read()
            with wave.open(path, "rb") as wf:
                sample_rate = wf.getframerate()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        latency_ms = (time.perf_counter() - t0) * 1000
        return TTSResult(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            engine=self.name,
            active_provider="CPUExecutionProvider",  # honest: pyttsx3 never uses NPU/GPU
            latency_ms=round(latency_ms, 3),
        )
