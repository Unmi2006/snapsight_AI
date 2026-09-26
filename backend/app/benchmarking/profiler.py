"""
Benchmark profiler: wraps inference calls and records real, measured
numbers -- never estimated or hardcoded.

Usage:
    with profile_call("whisper_base_en") as p:
        result = engine.run(handle, feeds)
    p.record["latency_ms"] is already filled by the context manager,
    but you can also read it directly from `result`.
"""
from __future__ import annotations

import time
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

import psutil

DB_PATH = Path(__file__).resolve().parent.parent.parent / "model_weights" / "benchmarks.db"
DB_PATH.parent.mkdir(exist_ok=True)

_process = psutil.Process()


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            model_name TEXT,
            active_provider TEXT,
            latency_ms REAL,
            mem_delta_mb REAL,
            cpu_percent REAL,
            metadata TEXT
        )
        """
    )
    conn.commit()
    conn.close()


_init_db()


@dataclass
class BenchmarkRecord:
    timestamp: float
    model_name: str
    active_provider: str
    latency_ms: float
    mem_delta_mb: float
    cpu_percent: float
    metadata: dict


def save_record(record: BenchmarkRecord) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO benchmark_runs
            (timestamp, model_name, active_provider, latency_ms, mem_delta_mb, cpu_percent, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.timestamp,
            record.model_name,
            record.active_provider,
            record.latency_ms,
            record.mem_delta_mb,
            record.cpu_percent,
            json.dumps(record.metadata),
        ),
    )
    conn.commit()
    conn.close()


def query_records(model_name: str | None = None, limit: int = 200) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if model_name:
        cur = conn.execute(
            "SELECT * FROM benchmark_runs WHERE model_name = ? ORDER BY id DESC LIMIT ?",
            (model_name, limit),
        )
    else:
        cur = conn.execute("SELECT * FROM benchmark_runs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def summary_stats(model_name: str | None = None) -> dict:
    rows = query_records(model_name=model_name, limit=5000)
    if not rows:
        return {"count": 0}
    latencies = [r["latency_ms"] for r in rows]
    return {
        "count": len(rows),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "providers_seen": sorted({r["active_provider"] for r in rows}),
    }


@contextmanager
def profile_call(model_name: str, active_provider: str = "unknown", metadata: dict | None = None):
    """
    Context manager: measures wall-clock latency, RSS memory delta, and
    process CPU% around whatever inference call happens inside the block.
    Automatically persists the record to the local SQLite store.
    """
    mem_before = _process.memory_info().rss / (1024 * 1024)
    _process.cpu_percent(interval=None)  # prime the counter
    t0 = time.perf_counter()
    try:
        yield
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000
        mem_after = _process.memory_info().rss / (1024 * 1024)
        cpu_percent = _process.cpu_percent(interval=None)
        record = BenchmarkRecord(
            timestamp=time.time(),
            model_name=model_name,
            active_provider=active_provider,
            latency_ms=round(latency_ms, 3),
            mem_delta_mb=round(mem_after - mem_before, 3),
            cpu_percent=cpu_percent,
            metadata=metadata or {},
        )
        save_record(record)
