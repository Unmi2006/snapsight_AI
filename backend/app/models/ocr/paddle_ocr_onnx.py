"""
ONNX-based OCR, routed through InferenceEngine so it honestly reports
whether it ran on NPU/GPU/CPU.

IMPORTANT / HONEST LIMITATION:
This backend needs `rec.onnx` (a CRNN-style text recognizer, e.g. an
exported PP-OCRv4 recognition model) placed at model_weights/ocr/rec.onnx.
This repo does NOT ship that file -- Qualcomm/PaddleOCR's pretrained
weights live on hosts outside what this dev sandbox can reach, so it must
be exported on YOUR machine. See scripts/export_ocr_model.py for exact
steps.

Text-region detection here is a deliberately simple OpenCV contour-based
proposal step, NOT the full PP-OCRv4 DB detector. It's good enough for
fairly clean printed textbook pages/code/equations, but a real DB detector
export is listed as a follow-up in the export script. Don't oversell this
as "PaddleOCR" in the demo -- describe it accurately: "ONNX CRNN
recognizer with a lightweight region proposal step, NPU-accelerated where
supported."
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.config import MODEL_WEIGHTS_DIR
from app.core.inference_engine import InferenceEngine
from app.models.ocr.base import OCRResult, TextBox

REC_MODEL_PATH = MODEL_WEIGHTS_DIR / "ocr" / "rec.onnx"
DICT_PATH = MODEL_WEIGHTS_DIR / "ocr" / "en_dict.txt"

REC_INPUT_HEIGHT = 48  # PP-OCRv4 rec model expects input shape [1, 3, 48, 320]


class PaddleOCRONNX:
    name = "paddle_ocr_onnx"

    def __init__(self, engine: InferenceEngine):
        if not REC_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{REC_MODEL_PATH} not found. Run "
                f"`python scripts/export_ocr_model.py` on a machine with "
                f"internet access to Qualcomm AI Hub / PaddleOCR's model "
                f"zoo first."
            )
        self.handle = engine.load(REC_MODEL_PATH, model_name="ocr_rec")
        self.chars = self._load_dict()

    @staticmethod
    def is_available() -> bool:
        return REC_MODEL_PATH.exists()

    def _load_dict(self) -> list[str]:
        if not DICT_PATH.exists():
            raise FileNotFoundError(f"Character dictionary missing at {DICT_PATH}")
        with open(DICT_PATH, encoding="utf-8") as f:
            chars = [line.rstrip("\n") for line in f]
        # index 0 is reserved for the CTC "blank" token
        return ["<blank>"] + chars

    def _propose_text_regions(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Simplified stand-in for a real detector: threshold + morphological
        close to merge letters into line-like blobs, then contour bounding
        boxes. Works reasonably for clean printed text on a plain
        background; will miss/merge lines on busy diagrams. Replace with an
        exported PP-OCRv4 'det.onnx' for production-grade detection.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        h_img, w_img = gray.shape
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 10 or h < 8:
                continue
            if w > 0.98 * w_img and h > 0.98 * h_img:
                continue  # whole-page box, not useful
            boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right
        return boxes

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        new_w = max(1, int(w * (REC_INPUT_HEIGHT / h)))
        resized = cv2.resize(crop_bgr, (new_w, REC_INPUT_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb / 255.0 - 0.5) / 0.5  # normalize to [-1, 1], standard for PP-OCR rec
        chw = rgb.transpose(2, 0, 1)
        return np.expand_dims(chw, axis=0)  # NCHW, N=1

    def _ctc_greedy_decode(self, logits: np.ndarray) -> tuple[str, float]:
        # logits shape: [1, T, num_classes]
        seq = logits[0]
        pred_ids = seq.argmax(axis=1)
        confidences = seq.max(axis=1)
        out_chars = []
        out_confs = []
        prev = -1
        for idx, conf in zip(pred_ids, confidences):
            if idx != 0 and idx != prev:  # skip blank + collapse repeats
                if idx < len(self.chars):
                    out_chars.append(self.chars[idx])
                    out_confs.append(conf)
            prev = idx
        text = "".join(out_chars)
        avg_conf = float(np.mean(out_confs)) * 100 if out_confs else 0.0
        return text, avg_conf

    def run(self, image: Image.Image) -> OCRResult:
        import time

        t0 = time.perf_counter()
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        region_boxes = self._propose_text_regions(bgr)

        boxes: list[TextBox] = []
        words: list[str] = []
        active_provider = self.handle.load_result.active_provider

        for (x, y, w, h) in region_boxes:
            crop = bgr[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            inp = self._preprocess_crop(crop)
            feeds = {self.handle.input_names[0]: inp.astype(np.float32)}
            outputs = self.handle.session.run(self.handle.output_names, feeds)
            text, conf = self._ctc_greedy_decode(outputs[0])
            if not text.strip():
                continue
            words.append(text)
            boxes.append(TextBox(text=text, confidence=conf, left=x, top=y, width=w, height=h))

        latency_ms = (time.perf_counter() - t0) * 1000
        return OCRResult(
            full_text=" ".join(words),
            boxes=boxes,
            engine=self.name,
            active_provider=active_provider,
            latency_ms=round(latency_ms, 3),
        )
