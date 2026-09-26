from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.utils.document_loader import load_document
from app.models.ocr.ocr_service import ocr_service

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_FILE_SIZE_MB = 25


@router.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Limit is {MAX_FILE_SIZE_MB} MB.")

    try:
        doc = load_document(file.filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if doc.kind == "pdf_digital":
        return {
            "kind": doc.kind,
            "page_count": doc.page_count,
            "text": doc.digital_text,
            "ocr_used": False,
        }

    # pdf_scanned or image -- run OCR page by page
    if not ocr_service.is_available:
        raise HTTPException(503, ocr_service.unavailable_reason or "OCR is not available.")

    page_results = []
    for page_image in doc.page_images:
        result = ocr_service.run(page_image)
        page_results.append(
            {
                "text": result.full_text,
                "engine": result.engine,
                "active_provider": result.active_provider,
                "latency_ms": result.latency_ms,
                "box_count": len(result.boxes),
            }
        )

    return {
        "kind": doc.kind,
        "page_count": doc.page_count,
        "text": "\n\n".join(p["text"] for p in page_results),
        "ocr_used": True,
        "pages": page_results,
    }
