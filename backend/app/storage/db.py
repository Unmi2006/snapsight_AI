"""
Shared SQLite plumbing for Phase 4 (Conversation History + Settings).

Deliberately a separate file from benchmarking/profiler.py's DB -- that
one is a high-frequency, append-only measurement log; this one is a much
lower-frequency store of user-facing state (conversation turns, settings)
that also supports UPDATE/DELETE. Keeping them apart means a bug or lock
contention in one never touches the other.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "model_weights" / "app_state.db"
DB_PATH.parent.mkdir(exist_ok=True)


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            kind TEXT NOT NULL,             -- "study_vision" | "voice"
            mode TEXT,                      -- study vision mode id, or "voice_ask"
            content_type TEXT,
            input_text TEXT,                -- OCR text or speech transcript
            question TEXT,                  -- custom/spoken question, if any
            answer_available INTEGER NOT NULL DEFAULT 0,
            answer_text TEXT,
            answer_engine TEXT,
            answer_provider TEXT,
            answer_latency_ms REAL,
            metadata TEXT                   -- JSON blob: full per-stage engine/provider/latency detail
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_kind_id ON conversations (kind, id)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


_init_db()
