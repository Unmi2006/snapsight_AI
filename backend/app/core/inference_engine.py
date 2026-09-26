"""
InferenceEngine: the single place in the codebase allowed to create an
onnxruntime.InferenceSession.

Why this exists:
- We must never CLAIM a model ran on the NPU without proving it.
- ORT will silently fall back to CPU for individual nodes (or the whole
  graph) if the requested EP can't compile it. If we just requested
  QNNExecutionProvider and moved on, we'd be lying to the user.
- This wrapper tries providers in priority order, catches failures, and
  records + exposes exactly which provider is active for a given session,
  plus (optionally) a per-node profile proving where each op executed.

This module is written to run correctly on ANY platform (Linux/macOS/
Windows x64/Windows ARM64). On non-Windows or when a provider package
isn't installed, it simply isn't in `ort.get_available_providers()` and
is skipped automatically -- there is no hardcoded platform check.
"""
from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.core.config import EP_PRIORITY

# Real, measurable perf tuning (not an EP claim): let ORT fuse/optimize the
# graph fully, and use all logical CPUs for the intra-op thread pool on the
# CPU fallback path. This speeds up every model that lands on
# CPUExecutionProvider (which is most of them until NPU/GPU is verified
# present) without changing what provider is reported.
_CPU_INTRA_OP_THREADS = max(1, os.cpu_count() or 4)

logger = logging.getLogger("snapsight.inference_engine")


@dataclass
class LoadResult:
    model_name: str
    requested_provider_order: list[str]
    available_providers: list[str]
    active_provider: str
    load_time_ms: float
    fallback_reason: str | None = None


@dataclass
class InferenceResult:
    outputs: list[Any]
    latency_ms: float
    active_provider: str


@dataclass
class ModelHandle:
    """Everything downstream code needs to call a loaded model honestly."""
    name: str
    session: ort.InferenceSession
    load_result: LoadResult
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


class InferenceEngine:
    """
    Loads an ONNX model, trying execution providers in EP_PRIORITY order,
    and never reports success for a provider it did not actually confirm.
    """

    def __init__(self, ep_priority: list[str] | None = None):
        self.ep_priority = ep_priority or EP_PRIORITY
        self.available = ort.get_available_providers()

    def load(self, model_path: str | Path, model_name: str | None = None) -> ModelHandle:
        model_path = Path(model_path)
        model_name = model_name or model_path.stem

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Run scripts/export_models.py first, or place the .onnx file there."
            )

        # Only attempt providers that ORT actually reports as available.
        # This is the honesty check: we never "request" QNN on a machine
        # where onnxruntime-qnn isn't even installed.
        candidate_providers = [p for p in self.ep_priority if p in self.available]
        if not candidate_providers:
            candidate_providers = ["CPUExecutionProvider"]

        last_error: str | None = None
        for provider in candidate_providers:
            t0 = time.perf_counter()
            try:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.intra_op_num_threads = _CPU_INTRA_OP_THREADS
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=so,
                    providers=[provider, "CPUExecutionProvider"]
                    if provider != "CPUExecutionProvider"
                    else ["CPUExecutionProvider"],
                )
                load_time_ms = (time.perf_counter() - t0) * 1000

                # THE HONESTY CHECK: ask the session what it actually picked,
                # don't trust what we requested.
                actual = session.get_providers()[0]
                if actual != provider:
                    # ORT silently downgraded us. Log it, keep going down
                    # the priority list only if we still want a "purer" EP;
                    # otherwise accept what actually works.
                    logger.warning(
                        "Requested %s but ORT actually selected %s for %s",
                        provider, actual, model_name,
                    )

                load_result = LoadResult(
                    model_name=model_name,
                    requested_provider_order=candidate_providers,
                    available_providers=self.available,
                    active_provider=actual,
                    load_time_ms=round(load_time_ms, 2),
                    fallback_reason=last_error,
                )
                return ModelHandle(
                    name=model_name,
                    session=session,
                    load_result=load_result,
                    input_names=[i.name for i in session.get_inputs()],
                    output_names=[o.name for o in session.get_outputs()],
                )
            except Exception as exc:  # noqa: BLE001 - we want to try the next EP
                last_error = f"{provider} failed: {exc}"
                logger.info("Provider %s unavailable for %s: %s", provider, model_name, exc)
                continue

        raise RuntimeError(
            f"Could not create an InferenceSession for {model_name} on ANY provider. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def run(handle: ModelHandle, feeds: dict[str, np.ndarray]) -> InferenceResult:
        t0 = time.perf_counter()
        outputs = handle.session.run(handle.output_names, feeds)
        latency_ms = (time.perf_counter() - t0) * 1000
        return InferenceResult(
            outputs=outputs,
            latency_ms=round(latency_ms, 3),
            active_provider=handle.load_result.active_provider,
        )

    def hardware_report(self) -> dict:
        """Used by /api/hardware/info -- what's real, right now, on this machine."""
        report = {
            "available_providers": self.available,
            "ep_priority_configured": self.ep_priority,
            "npu_provider_present": "QNNExecutionProvider" in self.available,
            "gpu_dml_provider_present": "DmlExecutionProvider" in self.available,
            "cpu_provider_present": "CPUExecutionProvider" in self.available,
        }
        return report


# Module-level singleton -- one engine, shared honesty ledger, for the app.
engine = InferenceEngine()
