"""
Tesseract-based OCR. This is deliberately NOT an NPU path -- Tesseract is a
classical (non-neural-network-on-ONNX) OCR engine, so it always runs on
CPU and we label it that way everywhere. It exists so the OCR feature
works end-to-end today, on any machine, with zero model downloads, while
the ONNX/NPU-accelerated path (paddle_ocr_onnx.py) is being set up.
"""
from __future__ import annotations

import os
import shutil
import time

import pytesseract
from PIL import Image

from app.models.ocr.base import OCRResult, TextBox

# Common install locations, checked if the binary isn't already on PATH.
# An explicit TESSERACT_CMD env var always wins.
_WINDOWS_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _resolve_tesseract_cmd() -> str | None:
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.isfile(env_path):
        return env_path

    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    for candidate in _WINDOWS_FALLBACK_PATHS:
        if os.path.isfile(candidate):
            return candidate

    return None


class TesseractOCR:
    name = "tesseract"

    def __init__(self):
        # Resolve the binary ourselves instead of trusting PATH, since PATH
        # edits on Windows often don't take effect in the shell/IDE that's
        # actually running uvicorn.
        resolved = _resolve_tesseract_cmd()
        if resolved:
            pytesseract.pytesseract.tesseract_cmd = resolved

        # Fail fast and clearly if the tesseract binary isn't installed,
        # rather than a cryptic error on first real request.
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Tesseract binary not found. Checked PATH, the TESSERACT_CMD "
                "env var, and the default Windows install location "
                r"(C:\Program Files\Tesseract-OCR\tesseract.exe). "
                "Install it: `apt install tesseract-ocr` (Linux) or "
                "https://github.com/UB-Mannheim/tesseract/wiki (Windows), "
                "or set TESSERACT_CMD to the full path of tesseract.exe."
            ) from exc

    def run(self, image: Image.Image) -> OCRResult:
        t0 = time.perf_counter()
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        latency_ms = (time.perf_counter() - t0) * 1000

        boxes: list[TextBox] = []
        words: list[str] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            if not text:
                continue
            conf_raw = data["conf"][i]
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            words.append(text)
            boxes.append(
                TextBox(
                    text=text,
                    confidence=conf,
                    left=data["left"][i],
                    top=data["top"][i],
                    width=data["width"][i],
                    height=data["height"][i],
                )
            )

        return OCRResult(
            full_text=" ".join(words),
            boxes=boxes,
            engine=self.name,
            active_provider="CPUExecutionProvider",  # honest: Tesseract never uses NPU/GPU
            latency_ms=round(latency_ms, 3),
        )
