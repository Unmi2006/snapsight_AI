"""
Whisper speech-to-text on raw ONNX Runtime via InferenceEngine -- the
NPU/GPU/CPU-capable STT path, same honesty pattern as OCR's
paddle_ocr_onnx.py: InferenceEngine walks EP_PRIORITY and reports
whichever provider ACTUALLY loaded the session (verified via
session.get_providers()), never just what was requested.

Expects a folder (model_weights/stt/) produced by
`optimum-cli export onnx --model openai/whisper-tiny.en
 --task automatic-speech-recognition` containing at minimum:
    encoder_model.onnx
    decoder_model.onnx     (the NON-past-key-values variant)
    tokenizer.json / vocab.json / merges.txt / tokenizer_config.json

See scripts/export_stt_model.py for the exact commands.

DOCUMENTED SIMPLIFICATIONS (read before trusting latency numbers):
1. No KV-cache decode loop. This backend re-feeds the entire token
   sequence generated so far on every decoding step and reads the logits
   at the last position, instead of wiring up the
   decoder_with_past_model.onnx / decoder_model_merged.onnx cache
   tensors. That's O(n^2) instead of O(n) token generation -- fine for
   short spoken questions on a tiny/base model, a real cost for long
   audio. A KV-cache decoder is the documented follow-up.
2. Mel filterbank is librosa's standard implementation, not OpenAI
   Whisper's exact shipped filter matrix -- very close, not
   bit-identical. Good enough for an MVP demo, not claimed to be more.
3. English-only in this MVP (forces
   <|startoftranscript|><|en|><|transcribe|><|notimestamps|>), matching
   the OCR module's en_dict.txt scope.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.inference_engine import InferenceEngine
from app.models.stt.base import STTResult

SAMPLE_RATE = 16000
N_MELS_DEFAULT = 80
MAX_AUDIO_SECONDS = 30  # Whisper's fixed context window
MAX_NEW_TOKENS = 200


class OnnxWhisperSTT:
    name = "whisper_onnx"

    def __init__(self, model_dir: str | Path, engine: InferenceEngine):
        from transformers import WhisperTokenizerFast

        self.model_dir = Path(model_dir)
        encoder_path = self.model_dir / "encoder_model.onnx"
        decoder_path = self.model_dir / "decoder_model.onnx"
        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(
                f"Expected encoder_model.onnx and decoder_model.onnx in {self.model_dir}"
            )

        self._engine = engine
        self._encoder = engine.load(encoder_path, model_name="whisper_encoder")
        self._decoder = engine.load(decoder_path, model_name="whisper_decoder")
        # Both halves are loaded through the same InferenceEngine instance
        # and, in practice, land on the same provider -- report the
        # encoder's verified provider as representative of the pair.
        self.active_provider = self._encoder.load_result.active_provider

        # Tokenizer files are read from the LOCAL export folder only --
        # no network call, same local-first rule as everywhere else here.
        self._tokenizer = WhisperTokenizerFast.from_pretrained(str(self.model_dir))
        self._sot = self._tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self._en = self._tokenizer.convert_tokens_to_ids("<|en|>")
        self._transcribe = self._tokenizer.convert_tokens_to_ids("<|transcribe|>")
        self._notimestamps = self._tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self._eot = self._tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self._prompt_len = 4  # sot, en, transcribe, notimestamps

    @staticmethod
    def is_available(model_dir: str | Path) -> bool:
        model_dir = Path(model_dir)
        if not (model_dir / "encoder_model.onnx").exists():
            return False
        if not (model_dir / "decoder_model.onnx").exists():
            return False
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _log_mel_spectrogram(self, samples: np.ndarray) -> np.ndarray:
        import librosa

        target_len = SAMPLE_RATE * MAX_AUDIO_SECONDS
        if len(samples) > target_len:
            samples = samples[:target_len]
        elif len(samples) < target_len:
            samples = np.pad(samples, (0, target_len - len(samples)))

        mel = librosa.feature.melspectrogram(
            y=samples.astype(np.float32),
            sr=SAMPLE_RATE,
            n_fft=400,
            hop_length=160,
            n_mels=N_MELS_DEFAULT,
        )
        log_mel = np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel[np.newaxis, :, :].astype(np.float32)  # [1, n_mels, T]

    def run(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> STTResult:
        t0 = time.perf_counter()
        mel = self._log_mel_spectrogram(samples)

        enc_result = self._engine.run(self._encoder, {"input_features": mel})
        encoder_hidden_states = enc_result.outputs[0]

        tokens = [self._sot, self._en, self._transcribe, self._notimestamps]
        for _ in range(MAX_NEW_TOKENS):
            input_ids = np.array([tokens], dtype=np.int64)
            dec_result = self._engine.run(
                self._decoder,
                {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states},
            )
            logits = dec_result.outputs[0]  # [1, seq_len, vocab]
            next_token = int(np.argmax(logits[0, -1]))
            if next_token == self._eot:
                break
            tokens.append(next_token)

        latency_ms = (time.perf_counter() - t0) * 1000
        text = self._tokenizer.decode(tokens[self._prompt_len :], skip_special_tokens=True).strip()

        return STTResult(
            text=text,
            engine=self.name,
            active_provider=self.active_provider,
            latency_ms=round(latency_ms, 3),
        )
