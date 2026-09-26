from __future__ import annotations

from fastapi import APIRouter, Query

from app.benchmarking.profiler import query_records, summary_stats

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


@router.get("/records")
def get_records(model_name: str | None = None, limit: int = Query(200, le=5000)):
    return {"records": query_records(model_name=model_name, limit=limit)}


@router.get("/summary")
def get_summary(model_name: str | None = None):
    return summary_stats(model_name=model_name)
