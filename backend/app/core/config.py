"""
Central configuration for SnapSight AI backend.

Nothing here talks to the network by design (local-first constraint).
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_WEIGHTS_DIR = BASE_DIR / "model_weights"
MODEL_WEIGHTS_DIR.mkdir(exist_ok=True)

# Phase 2: local LLM (Study Vision reasoning). Each exported model variant
# lives in its own subfolder because onnxruntime-genai bakes the execution
# provider into the export itself (unlike raw ONNX Runtime, there's no
# single file that can be retargeted at load time -- see
# app/models/llm/onnx_genai_backend.py). LLMService walks this list top to
# bottom and uses the first folder that exists AND whose backing package
# is importable -- the LLM equivalent of EP_PRIORITY above.
LLM_MODELS_DIR = MODEL_WEIGHTS_DIR / "llm"
LLM_MODELS_DIR.mkdir(exist_ok=True)

LLM_EP_PRIORITY = [
    ("qnn", "QNNExecutionProvider"),   # Snapdragon Hexagon NPU
    ("dml", "DmlExecutionProvider"),   # DirectML GPU
    ("cpu", "CPUExecutionProvider"),   # Always-installable fallback
]

# Execution provider priority order. The InferenceEngine walks this list
# top to bottom and uses the FIRST provider that (a) is installed and
# (b) successfully creates an InferenceSession for the given model.
# This is intentionally explicit rather than "just trust ORT" because we
# need to KNOW and REPORT which one actually ran.
EP_PRIORITY = [
    "QNNExecutionProvider",   # Snapdragon Hexagon NPU (Windows ARM64 only)
    "DmlExecutionProvider",   # DirectML GPU (Windows x64/ARM64)
    "CPUExecutionProvider",   # Always available, always last resort
]

# QNN EP requires a backend library path on Windows. This is only used
# if QNNExecutionProvider is actually present in ort.get_available_providers().
QNN_BACKEND_PATH = "QnnHtp.dll"

# Phase 3: local speech-to-text (Voice Assistant). A single folder, like
# OCR's model_weights/ocr/ -- unlike the LLM's per-EP qnn/dml/cpu
# subfolders, a plain ONNX Whisper export isn't baked to one EP at export
# time, so InferenceEngine's own provider selection handles NPU/GPU/CPU
# from this one folder.
STT_MODEL_DIR = MODEL_WEIGHTS_DIR / "stt"
STT_MODEL_DIR.mkdir(exist_ok=True)

# Phase 3: local text-to-speech (Voice Assistant). Same single-folder
# reasoning as STT_MODEL_DIR above -- Piper's ONNX voice files aren't
# baked to a specific EP at export time either.
TTS_MODEL_DIR = MODEL_WEIGHTS_DIR / "tts"
TTS_MODEL_DIR.mkdir(exist_ok=True)

APP_NAME = "SnapSight AI"
APP_VERSION = "0.1.0-mvp"

# Phase 5: production single-process serving. When frontend/dist exists
# (i.e. someone ran `npm run build`), main.py mounts and serves it
# directly so a demo is one process/one port instead of two dev servers.
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "dist"

CORS_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:5173",
]
