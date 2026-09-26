"""
/api/voice/* -- Phase 3: Voice Assistant. Speech in, local LLM reasoning
(reusing the same LLMService from Phase 2, and, when an image is
attached, the same Study Vision orchestrator too), speech out. Every
response reports which STT/LLM/TTS engine and execution provider
actually ran -- same honesty rule as every other endpoint in this
backend. If a stage is unavailable, that's reported plainly instead of
faking output, and earlier stages that did succeed are still returned.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import Response
from PIL import Image

from app.models.llm.base import GenerationConfig
from app.models.llm.llm_service import LLMUnavailableError, llm_service
from app.models.ocr.ocr_service import ocr_service
from app.models.stt.stt_service import stt_service
from app.models.tts.tts_service import tts_service
from app.orchestrator.voice_assistant import build_voice_prompt
from app.storage.conversation_store import ConversationRecord, save_conversation
from app.utils.audio_loader import load_audio

router = APIRouter(prefix="/api/voice", tags=["voice"])
logger = logging.getLogger("snapsight.api.voice")

MAX_NEW_TOKENS_CEILING = 1024


@router.get("/status")
def voice_status():
    """Drives the Voice Assistant page's availability banners for all
    three stages -- the frontend never assumes any of them are ready."""
    return {
        "stt": stt_service.status(),
        "tts": tts_service.status(),
        "llm": llm_service.status(),
    }


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Speech-to-text only -- lets the frontend show the transcript
    immediately, and is also useful standalone for testing the STT stage."""
    if not stt_service.is_available:
        raise HTTPException(503, stt_service.unavailable_reason or "STT is not available.")

    file_bytes = await file.read()
    try:
        samples = load_audio(file_bytes, file.filename or "")
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc

    result = stt_service.transcribe(samples)
    return {
        "text": result.text,
        "engine": result.engine,
        "active_provider": result.active_provider,
        "latency_ms": result.latency_ms,
    }


@router.post("/speak")
async def speak(text: str = Form(...)):
    """Text-to-speech only -- returns a playable WAV file directly.
    Useful standalone for testing the TTS stage."""
    if not text.strip():
        raise HTTPException(400, "`text` must be non-empty.")
    if not tts_service.is_available:
        raise HTTPException(503, tts_service.unavailable_reason or "TTS is not available.")

    result = tts_service.synthesize(text)
    headers = {
        "X-TTS-Engine": result.engine,
        "X-TTS-Provider": result.active_provider,
        "X-TTS-Latency-Ms": str(result.latency_ms),
        "X-Sample-Rate": str(result.sample_rate),
    }
    return Response(content=result.audio_bytes, media_type="audio/wav", headers=headers)


@router.post("/ask")
async def voice_ask(
    audio: UploadFile = File(...),
    image: Optional[UploadFile] = File(None),
    max_new_tokens: int = Form(300),
    speak_response: bool = Form(True),
):
    """
    The full Voice Assistant loop: transcribe the spoken question, run
    OCR on an optionally-attached image (reusing the exact same
    OCRService as Camera Mode / Study Vision), build a prompt via the
    orchestrator, reason with the local LLM, and (if requested) speak the
    answer back with TTS. Every stage degrades gracefully and reports
    itself honestly -- a missing LLM or TTS backend still returns the
    stages that did succeed instead of a 500.
    """
    if not stt_service.is_available:
        raise HTTPException(503, stt_service.unavailable_reason or "STT is not available.")

    audio_bytes = await audio.read()
    try:
        samples = load_audio(audio_bytes, audio.filename or "")
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc

    stt_result = stt_service.transcribe(samples)

    response: dict = {
        "stt": {
            "text": stt_result.text,
            "engine": stt_result.engine,
            "active_provider": stt_result.active_provider,
            "latency_ms": stt_result.latency_ms,
        }
    }

    ocr_text: Optional[str] = None
    if image is not None:
        image_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not decode image: {exc}") from exc

        if ocr_service.is_available:
            ocr_result = ocr_service.run(pil_image)
            ocr_text = ocr_result.full_text
            response["ocr"] = {
                "available": True,
                "text": ocr_result.full_text,
                "engine": ocr_result.engine,
                "active_provider": ocr_result.active_provider,
                "latency_ms": ocr_result.latency_ms,
            }
        else:
            response["ocr"] = {"available": False, "reason": ocr_service.unavailable_reason}

    try:
        prompt_text = build_voice_prompt(stt_result.text, ocr_text)
    except ValueError as exc:
        # A genuinely empty transcription -- return what we have (the STT
        # stage, and the OCR stage if an image was sent) instead of a 400
        # that discards everything, since silence/noise from a mic is a
        # normal occurrence, not a client error.
        response["llm"] = {"available": False, "reason": str(exc)}
        response["tts"] = {"available": False, "reason": "No question to answer."}
        return response

    if not llm_service.is_available:
        response["llm"] = {"available": False, "reason": llm_service.status()["reason"]}
        response["tts"] = {"available": False, "reason": "LLM unavailable -- nothing to speak."}
        _log_voice_conversation(stt_result.text, response)
        return response

    config = GenerationConfig(max_new_tokens=min(max_new_tokens, MAX_NEW_TOKENS_CEILING))
    try:
        llm_result = llm_service.generate(prompt_text, config)
    except LLMUnavailableError as exc:
        response["llm"] = {"available": False, "reason": str(exc)}
        response["tts"] = {"available": False, "reason": "LLM unavailable -- nothing to speak."}
        _log_voice_conversation(stt_result.text, response)
        return response

    response["llm"] = {
        "available": True,
        "answer": llm_result.text,
        "engine": llm_result.engine,
        "model_name": llm_result.model_name,
        "active_provider": llm_result.active_provider,
        "latency_ms": llm_result.latency_ms,
    }

    if speak_response and tts_service.is_available:
        tts_result = tts_service.synthesize(llm_result.text)
        response["tts"] = {
            "available": True,
            "audio_base64": base64.b64encode(tts_result.audio_bytes).decode("ascii"),
            "sample_rate": tts_result.sample_rate,
            "engine": tts_result.engine,
            "active_provider": tts_result.active_provider,
            "latency_ms": tts_result.latency_ms,
        }
    elif not speak_response:
        response["tts"] = {"available": False, "reason": "speak_response was set to false."}
    else:
        response["tts"] = {"available": False, "reason": tts_service.unavailable_reason}

    _log_voice_conversation(stt_result.text, response)
    return response


def _log_voice_conversation(transcript: str, response: dict) -> None:
    """Phase 4: save this Voice Assistant turn to Conversation History.
    Best-effort -- a storage hiccup should never break the actual
    feature, so failures are logged and swallowed, never raised. Audio
    (both the question and the spoken reply) is deliberately not
    persisted -- see conversation_store.py's docstring; History's
    "replay" re-synthesizes the saved answer text instead."""
    try:
        llm = response.get("llm", {})
        save_conversation(
            ConversationRecord(
                kind="voice",
                mode="voice_ask",
                content_type=None,
                input_text=transcript,
                question=transcript or None,
                answer_available=bool(llm.get("available")),
                answer_text=llm.get("answer"),
                answer_engine=llm.get("engine"),
                answer_provider=llm.get("active_provider"),
                answer_latency_ms=llm.get("latency_ms"),
                metadata={
                    "stt": response.get("stt", {}),
                    "ocr": response.get("ocr", {}),
                    "llm": llm,
                    "tts": {k: v for k, v in response.get("tts", {}).items() if k != "audio_base64"},
                },
            )
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save Voice Assistant turn to Conversation History.")
