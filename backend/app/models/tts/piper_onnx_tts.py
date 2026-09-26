"""
Neural text-to-speech via a Piper voice model on raw ONNX Runtime through
InferenceEngine -- the NPU/GPU/CPU-capable TTS path, same honesty pattern
as every other InferenceEngine consumer in this codebase: whichever
provider ACTUALLY loaded the session is what gets reported, verified via
session.get_providers(), never just "requested".

Expects, under model_weights/tts/:
    <voice>.onnx        -- the Piper VITS-style model (single file)
    <voice>.onnx.json   -- Piper's config: sample_rate, phoneme_id_map, espeak voice

Requires `piper-phonemize` (bundles espeak-ng) for grapheme-to-phoneme
conversion -- a real, separate native dependency, same category as
Tesseract/poppler/ffmpeg elsewhere in this repo. See
scripts/export_tts_model.py.
"""
from __future__ import annotations

import io
import json
import time
import wave
from pathlib import Path

import numpy as np

from app.core.inference_engine import InferenceEngine
from app.models.tts.base import TTSResult


class PiperOnnxTTS:
    name = "piper_onnx"

    def __init__(self, model_dir: str | Path, engine: InferenceEngine):
        self.model_dir = Path(model_dir)
        onnx_files = sorted(self.model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No Piper .onnx voice found in {self.model_dir}")
        self.voice_path = onnx_files[0]
        config_path = Path(str(self.voice_path) + ".json")
        if not config_path.exists():
            raise FileNotFoundError(f"Missing Piper voice config: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            self.config = json.load(f)
        self.sample_rate = self.config.get("audio", {}).get("sample_rate", 22050)
        self.phoneme_id_map: dict = self.config.get("phoneme_id_map", {})
        self.espeak_voice = self.config.get("espeak", {}).get("voice", "en-us")

        self._engine = engine
        self._handle = engine.load(self.voice_path, model_name="piper_tts")
        self.active_provider = self._handle.load_result.active_provider

    @staticmethod
    def is_available(model_dir: str | Path) -> bool:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            return False
        onnx_files = sorted(model_dir.glob("*.onnx"))
        if not onnx_files or not Path(str(onnx_files[0]) + ".json").exists():
            return False
        try:
            import piper_phonemize  # noqa: F401

            return True
        except ImportError:
            return False

    def _text_to_ids(self, text: str) -> np.ndarray:
        import piper_phonemize

        phoneme_sequences = piper_phonemize.phonemize_espeak(text, self.espeak_voice)
        pad_id = self.phoneme_id_map.get("_", [0])[0]
        ids: list[int] = [self.phoneme_id_map.get("^", [pad_id])[0]]
        for sentence in phoneme_sequences:
            for phoneme in sentence:
                mapped = self.phoneme_id_map.get(phoneme)
                if mapped:
                    ids.extend(mapped)
                    ids.append(pad_id)
        ids.append(self.phoneme_id_map.get("$", [pad_id])[0])
        return np.array([ids], dtype=np.int64)

    def run(self, text: str) -> TTSResult:
        t0 = time.perf_counter()
        ids = self._text_to_ids(text)
        lengths = np.array([ids.shape[1]], dtype=np.int64)
        # Piper's standard inference-time scales: [noise_scale, length_scale, noise_w].
        scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

        result = self._engine.run(
            self._handle,
            {"input": ids, "input_lengths": lengths, "scales": scales},
        )
        audio = np.asarray(result.outputs[0]).squeeze().astype(np.float32)
        audio_int16 = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        latency_ms = (time.perf_counter() - t0) * 1000
        return TTSResult(
            audio_bytes=buf.getvalue(),
            sample_rate=self.sample_rate,
            engine=self.name,
            active_provider=self.active_provider,
            latency_ms=round(latency_ms, 3),
        )
