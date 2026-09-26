"""
/api/history/* -- Phase 4: Conversation History.

Every completed Study Vision (`/api/vision/study`) and Voice Assistant
(`/api/voice/ask`) turn is saved by its own endpoint (see the
`_log_conversation` calls in vision.py / voice.py) into
app/storage/conversation_store.py. This router only reads, replays, and
deletes what's already there -- it never generates content itself.

"Replay" here is honest about what it actually is: raw audio isn't
persisted (see conversation_store.py's docstring for why), so replaying
a voice turn's answer means re-synthesizing the *saved answer text*
through whichever TTS backend is active right now -- not playing back
the original clip. The endpoint says so explicitly in its response.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.models.tts.tts_service import tts_service
from app.storage import conversation_store

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("/summary")
def get_summary():
    return conversation_store.stats()


@router.get("/list")
def list_history(
    kind: str | None = Query(None, pattern="^(study_vision|voice)$"),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
):
    return {"items": conversation_store.list_conversations(kind=kind, limit=limit, offset=offset)}


@router.get("/{conversation_id}")
def get_history_item(conversation_id: int):
    record = conversation_store.get_conversation(conversation_id)
    if record is None:
        raise HTTPException(404, f"No history entry with id {conversation_id}.")
    return record


@router.post("/{conversation_id}/replay-audio")
def replay_audio(conversation_id: int):
    """Re-synthesizes the saved answer text through the CURRENTLY ACTIVE
    TTS backend and returns a playable WAV -- same response shape as
    /api/voice/speak. Not a recording of the original spoken reply (none
    was kept); if the TTS backend or voice has changed since the turn
    was saved, this will sound different, and that's expected."""
    record = conversation_store.get_conversation(conversation_id)
    if record is None:
        raise HTTPException(404, f"No history entry with id {conversation_id}.")
    if not record["answer_available"] or not record.get("answer_text", "").strip():
        raise HTTPException(400, "This history entry has no saved answer text to speak.")
    if not tts_service.is_available:
        raise HTTPException(503, tts_service.unavailable_reason or "TTS is not available.")

    result = tts_service.synthesize(record["answer_text"])
    headers = {
        "X-TTS-Engine": result.engine,
        "X-TTS-Provider": result.active_provider,
        "X-TTS-Latency-Ms": str(result.latency_ms),
        "X-Sample-Rate": str(result.sample_rate),
        "X-Replay-Note": "resynthesized-from-saved-text",
    }
    return Response(content=result.audio_bytes, media_type="audio/wav", headers=headers)


@router.delete("/{conversation_id}")
def delete_history_item(conversation_id: int):
    deleted = conversation_store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(404, f"No history entry with id {conversation_id}.")
    return {"deleted": True, "id": conversation_id}


@router.delete("")
def clear_history(kind: str | None = Query(None, pattern="^(study_vision|voice)$")):
    count = conversation_store.clear_conversations(kind=kind)
    return {"deleted_count": count, "kind": kind}
