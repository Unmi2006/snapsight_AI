"""
Phase 4: Conversation History storage.

Every completed Study Vision request (app/api/vision.py) and Voice
Assistant `/ask` request (app/api/voice.py) gets one row here -- OCR/
transcript text in, LLM answer out (if the LLM was available), plus a
JSON blob of every stage's engine/provider/latency so the History page
can show exactly what actually ran, same honesty rule as everywhere else
in this backend.

Deliberately NOT stored: raw image bytes or raw audio bytes -- keeping
this a lightweight, purely-textual log rather than a growing media
archive. "Replay" (app/api/history.py) means re-showing the saved text,
or re-synthesizing the saved answer text through whichever TTS backend
is currently active -- it is honestly a *new* synthesis of old text, not
a recording of the original audio, and the API says so explicitly.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from app.storage.db import get_connection

PREVIEW_CHARS = 160


@dataclass
class ConversationRecord:
    kind: str                      # "study_vision" | "voice"
    mode: str | None = None
    content_type: str | None = None
    input_text: str | None = None
    question: str | None = None
    answer_available: bool = False
    answer_text: str | None = None
    answer_engine: str | None = None
    answer_provider: str | None = None
    answer_latency_ms: float | None = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


def save_conversation(record: ConversationRecord) -> int:
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO conversations
            (timestamp, kind, mode, content_type, input_text, question,
             answer_available, answer_text, answer_engine, answer_provider,
             answer_latency_ms, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.timestamp,
            record.kind,
            record.mode,
            record.content_type,
            record.input_text,
            record.question,
            int(record.answer_available),
            record.answer_text,
            record.answer_engine,
            record.answer_provider,
            record.answer_latency_ms,
            json.dumps(record.metadata),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def _preview(text: str | None) -> str | None:
    if not text:
        return text
    text = text.strip()
    if len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS].rstrip() + "…"


def list_conversations(kind: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    conn = get_connection()
    if kind:
        cur = conn.execute(
            """
            SELECT id, timestamp, kind, mode, content_type, input_text, question,
                   answer_available, answer_text
            FROM conversations WHERE kind = ? ORDER BY id DESC LIMIT ? OFFSET ?
            """,
            (kind, limit, offset),
        )
    else:
        cur = conn.execute(
            """
            SELECT id, timestamp, kind, mode, content_type, input_text, question,
                   answer_available, answer_text
            FROM conversations ORDER BY id DESC LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    for row in rows:
        row["input_preview"] = _preview(row.pop("input_text"))
        row["answer_preview"] = _preview(row.pop("answer_text"))
        row["answer_available"] = bool(row["answer_available"])
    return rows


def get_conversation(conversation_id: int) -> dict | None:
    conn = get_connection()
    cur = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    result = dict(row)
    result["answer_available"] = bool(result["answer_available"])
    try:
        result["metadata"] = json.loads(result["metadata"]) if result["metadata"] else {}
    except json.JSONDecodeError:
        result["metadata"] = {}
    return result


def delete_conversation(conversation_id: int) -> bool:
    conn = get_connection()
    cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def clear_conversations(kind: str | None = None) -> int:
    conn = get_connection()
    if kind:
        cur = conn.execute("DELETE FROM conversations WHERE kind = ?", (kind,))
    else:
        cur = conn.execute("DELETE FROM conversations")
    conn.commit()
    count = cur.rowcount
    conn.close()
    return count


def stats() -> dict:
    conn = get_connection()
    cur = conn.execute("SELECT kind, COUNT(*) as n FROM conversations GROUP BY kind")
    by_kind = {row["kind"]: row["n"] for row in cur.fetchall()}
    total = conn.execute("SELECT COUNT(*) as n FROM conversations").fetchone()["n"]
    with_answer = conn.execute(
        "SELECT COUNT(*) as n FROM conversations WHERE answer_available = 1"
    ).fetchone()["n"]
    conn.close()
    return {"total": total, "by_kind": by_kind, "with_answer": with_answer}
