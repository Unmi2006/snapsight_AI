from __future__ import annotations

import io

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from app.models.ocr.ocr_service import ocr_service

router = APIRouter(prefix="/api/camera", tags=["camera"])


@router.post("/capture")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Accepts one still frame captured by the browser (a JPEG/PNG blob from
    a <canvas>.toBlob() call on the client), runs OCR on it, and returns
    the extracted text plus bounding boxes so the frontend can draw an
    overlay. This is the building block Study Vision Mode (Phase 2) will
    call into with different prompt templates on top.
    """
    file_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode image: {exc}") from exc

    if not ocr_service.is_available:
        raise HTTPException(503, ocr_service.unavailable_reason or "OCR is not available.")

    result = ocr_service.run(image)
    return {
        "text": result.full_text,
        "engine": result.engine,
        "active_provider": result.active_provider,
        "latency_ms": result.latency_ms,
        "boxes": [
            {
                "text": b.text,
                "confidence": b.confidence,
                "left": b.left,
                "top": b.top,
                "width": b.width,
                "height": b.height,
            }
            for b in result.boxes
        ],
    }
