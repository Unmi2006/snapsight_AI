"""
/api/vision/* -- Study Vision Mode. Ties together the pieces built in
Phase 1 (OCR) and Phase 2 (LLM + orchestrator): a captured frame goes
through OCR first, the orchestrator turns the extracted text + the user's
chosen task into one prompt, then the local LLM (if installed) reasons
over it. Every response reports which OCR/LLM engine and execution
provider actually ran -- and if the LLM isn't installed yet, this says so
plainly instead of returning a fake answer. OCR results are still
returned even when the LLM is unavailable, so the feature degrades
gracefully instead of failing outright.
"""
from __future__ import annotations

import io
import logging

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from PIL import Image

from app.models.llm.base import GenerationConfig
from app.models.llm.llm_service import LLMUnavailableError, llm_service
from app.models.ocr.ocr_service import ocr_service
from app.orchestrator.study_vision import CONTENT_TYPES, build_prompt, list_modes
from app.storage.conversation_store import ConversationRecord, save_conversation

router = APIRouter(prefix="/api/vision", tags=["vision"])
logger = logging.getLogger("snapsight.api.vision")

MAX_NEW_TOKENS_CEILING = 1024


@router.get("/modes")
def get_modes():
    """Drives the mode/content-type buttons on the Study Vision page --
    the frontend never hardcodes this list, it asks the backend."""
    return {"modes": list_modes(), "content_types": CONTENT_TYPES}


@router.get("/llm-status")
def llm_status():
    return llm_service.status()


@router.post("/study")
async def study_vision(
    file: UploadFile = File(...),
    mode: str = Form("explain"),
    content_type: str = Form("auto"),
    question: str = Form(""),
    max_new_tokens: int = Form(400),
):
    file_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode image: {exc}") from exc

    if not ocr_service.is_available:
        raise HTTPException(503, ocr_service.unavailable_reason or "OCR is not available.")

    ocr_result = ocr_service.run(image)

    try:
        prompt = build_prompt(
            ocr_result.full_text,
            mode=mode,
            custom_question=question,
            content_type=content_type,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    response: dict = {
        "ocr": {
            "text": ocr_result.full_text,
            "engine": ocr_result.engine,
            "active_provider": ocr_result.active_provider,
            "latency_ms": ocr_result.latency_ms,
            "box_count": len(ocr_result.boxes),
        },
        "mode": mode,
        "content_type": content_type,
        "instruction": prompt.instruction,
    }

    if not llm_service.is_available:
        response["llm"] = {"available": False, "reason": llm_service.status()["reason"]}
        _log_conversation(mode, content_type, question, ocr_result.full_text, response)
        return response

    config = GenerationConfig(max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS_CEILING))
    try:
        llm_result = llm_service.generate(prompt.prompt_text, config)
    except LLMUnavailableError as exc:
        response["llm"] = {"available": False, "reason": str(exc)}
        _log_conversation(mode, content_type, question, ocr_result.full_text, response)
        return response

    response["llm"] = {
        "available": True,
        "answer": llm_result.text,
        "engine": llm_result.engine,
        "model_name": llm_result.model_name,
        "active_provider": llm_result.active_provider,
        "latency_ms": llm_result.latency_ms,
        "prompt_tokens": llm_result.prompt_tokens,
        "completion_tokens": llm_result.completion_tokens,
    }
    _log_conversation(mode, content_type, question, ocr_result.full_text, response)
    return response


def _log_conversation(mode: str, content_type: str, question: str, ocr_text: str, response: dict) -> None:
    """Phase 4: save this Study Vision turn to Conversation History.
    Best-effort -- a storage hiccup should never break the actual
    feature, so failures are logged and swallowed, never raised."""
    try:
        llm = response.get("llm", {})
        save_conversation(
            ConversationRecord(
                kind="study_vision",
                mode=mode,
                content_type=content_type,
                input_text=ocr_text,
                question=question or None,
                answer_available=bool(llm.get("available")),
                answer_text=llm.get("answer"),
                answer_engine=llm.get("engine"),
                answer_provider=llm.get("active_provider"),
                answer_latency_ms=llm.get("latency_ms"),
                metadata={"ocr": response.get("ocr", {}), "llm": llm},
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save Study Vision turn to Conversation History.")
