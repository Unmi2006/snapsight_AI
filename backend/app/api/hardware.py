"""
/api/hardware/* -- the endpoints that back the "Hardware/Runtime Info" UI
page. Everything here is measured live, not hardcoded, so the badge the
frontend shows ("NPU: verified" vs "NPU: not detected") is always true
for the machine actually running the backend right now.
"""
from __future__ import annotations

import platform
import time

import numpy as np
import onnxruntime as ort
import psutil
from fastapi import APIRouter

from app.core.inference_engine import engine

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


def _micro_benchmark_provider(provider: str) -> dict:
    """
    Runs a tiny matmul ONNX graph on the given provider and reports whether
    it actually initialized on that provider and how long a trivial op took.
    This is what lets the UI say "verified" instead of "assumed".
    """
    try:
        import onnx
        from onnx import helper, TensorProto

        # Build a minimal 64x64 matmul graph in-memory -- no file needed.
        A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [64, 64])
        B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [64, 64])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [64, 64])
        node = helper.make_node("MatMul", ["A", "B"], ["Y"])
        graph = helper.make_graph([node], "micro_bench", [A, B], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        # Newer `onnx` package versions default to a higher IR version than
        # some `onnxruntime` builds support. Pin it down explicitly so this
        # micro-benchmark doesn't break just because onnx got upgraded.
        model.ir_version = 10

        providers = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
        t0 = time.perf_counter()
        sess = ort.InferenceSession(model.SerializeToString(), providers=providers)
        load_ms = (time.perf_counter() - t0) * 1000

        actual = sess.get_providers()[0]
        a = np.random.rand(64, 64).astype(np.float32)
        b = np.random.rand(64, 64).astype(np.float32)
        t0 = time.perf_counter()
        sess.run(None, {"A": a, "B": b})
        run_ms = (time.perf_counter() - t0) * 1000

        return {
            "requested": provider,
            "actually_used": actual,
            "verified": actual == provider,
            "session_load_ms": round(load_ms, 3),
            "op_run_ms": round(run_ms, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {"requested": provider, "actually_used": None, "verified": False, "error": str(exc)}


@router.get("/info")
def hardware_info():
    report = engine.hardware_report()
    system = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }
    return {"system": system, **report}


@router.get("/verify")
def verify_providers():
    """
    Live micro-benchmark of every ORT-available provider on this machine.
    This is the endpoint the Hardware Info page polls to render honest
    NPU/GPU/CPU badges -- 'verified' means we actually ran an op on it,
    not that we merely requested it.
    """
    results = [_micro_benchmark_provider(p) for p in ort.get_available_providers()]
    return {"results": results}
