"""
Turns an uploaded file (PDF or image) into something the OCR/vision layer
can work with: either extracted digital text (fast path for born-digital
PDFs) or a list of PIL images to run OCR on (scanned PDFs, photos).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

from pdf2image import convert_from_bytes
from PIL import Image
from pypdf import PdfReader

MAX_PDF_PAGES = 20  # sane ceiling so a huge PDF can't hang the demo


@dataclass
class LoadedDocument:
    kind: str                      # "pdf_digital" | "pdf_scanned" | "image"
    digital_text: str = ""         # populated for pdf_digital
    page_images: list[Image.Image] = field(default_factory=list)  # populated for pdf_scanned / image
    page_count: int = 0


def load_pdf(file_bytes: bytes) -> LoadedDocument:
    reader = PdfReader(io.BytesIO(file_bytes))
    page_count = len(reader.pages)
    text_chunks = [page.extract_text() or "" for page in reader.pages[:MAX_PDF_PAGES]]
    combined = "\n".join(t.strip() for t in text_chunks if t.strip())

    # Heuristic: if we got a meaningful amount of embedded text, treat this
    # as a "digital" PDF and skip OCR entirely -- much faster and more
    # accurate than re-OCRing text that's already in the file.
    if len(combined) > 40:
        return LoadedDocument(kind="pdf_digital", digital_text=combined, page_count=page_count)

    # Otherwise assume it's scanned/image-based and rasterize pages for OCR.
    images = convert_from_bytes(file_bytes, dpi=200)[:MAX_PDF_PAGES]
    return LoadedDocument(kind="pdf_scanned", page_images=images, page_count=page_count)


def load_image(file_bytes: bytes) -> LoadedDocument:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return LoadedDocument(kind="image", page_images=[image], page_count=1)


def load_document(filename: str, file_bytes: bytes) -> LoadedDocument:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return load_pdf(file_bytes)
    if lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        return load_image(file_bytes)
    raise ValueError(f"Unsupported file type: {filename}. Use PDF, PNG, JPG, BMP, or WEBP.")
