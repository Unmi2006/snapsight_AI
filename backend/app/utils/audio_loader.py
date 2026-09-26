"""
Decodes an uploaded audio blob -- whatever container/codec the browser's
MediaRecorder produced (typically webm/opus in Chrome/Edge, ogg/opus in
Firefox), or a plain .wav from a test script -- into a mono float32 numpy
array at 16kHz, the format both STT backends expect.

Requires ffmpeg on PATH (via pydub). This is the same category of
external-binary dependency as Tesseract/poppler in Phase 1 -- not a
Python-only dependency, install it separately.
"""
from __future__ import annotations

import io

import numpy as np

TARGET_SAMPLE_RATE = 16000


def load_audio(file_bytes: bytes, filename: str = "") -> np.ndarray:
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError(
            "pydub is not installed, so uploaded audio can't be decoded. "
            "Install it with `pip install pydub` (and make sure the ffmpeg "
            "binary is on PATH -- `apt install ffmpeg` on Linux, `brew install "
            "ffmpeg` on macOS)."
        ) from exc

    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not decode audio ({exc}). This usually means ffmpeg isn't "
            "installed / on PATH -- install it: `apt install ffmpeg` (Linux), "
            "`brew install ffmpeg` (macOS), or download a build and add it to "
            "PATH (Windows)."
        ) from exc

    audio = audio.set_channels(1).set_frame_rate(TARGET_SAMPLE_RATE)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    if samples.size == 0:
        raise RuntimeError("Decoded audio has zero samples -- the recording may be empty or corrupted.")

    max_val = float(1 << (8 * audio.sample_width - 1))
    samples = samples / max_val
    return samples
