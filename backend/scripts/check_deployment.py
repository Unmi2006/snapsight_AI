"""
Phase 5: deployment preflight check.

Run this BEFORE a demo, on whatever machine you're about to demo on:

    cd backend
    python scripts/check_deployment.py

It answers one question honestly: "if I start the server right now, what
will actually work?" -- same anti-fake-NPU-claim philosophy as the rest
of this backend, just as a fast, no-server-needed CLI instead of an API
call. Nothing here is a live inference test (that's what
`/api/hardware/verify` already does once the server is up) -- this is
the layer BELOW that: is the interpreter itself the right one, are the
system binaries on PATH, are the packages importable, are any models
actually exported yet.

Exit codes (useful in a setup script): 0 = fully ready, 1 = ready with
warnings (fallback-only paths), 2 = a blocking problem was found.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import struct
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

STATUS_OK, STATUS_WARN, STATUS_FAIL = "OK", "WARN", "FAIL"
_ICON = {STATUS_OK: "[ OK ]", STATUS_WARN: "[WARN]", STATUS_FAIL: "[FAIL]"}

results: list[tuple[str, str, str]] = []  # (section, status, message)


def check(section: str, status: str, message: str) -> None:
    results.append((section, status, message))


def _module_importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# --------------------------------------------------------------------------
# 1. Platform / interpreter -- catches the single most common Snapdragon
#    deployment mistake: an x64 Python running under Windows' ARM64
#    emulation layer. It IMPORTS fine and RUNS fine -- it just silently
#    can never load QNNExecutionProvider, which looks identical to "NPU
#    not supported" unless you check this specifically.
# --------------------------------------------------------------------------
system = platform.system()
machine = platform.machine()
bitness = struct.calcsize("P") * 8
emulated_hint = os.environ.get("PROCESSOR_ARCHITEW6432")  # set by WOW64 on Windows

check("Platform", STATUS_OK, f"{system} / {machine} / {bitness}-bit Python {platform.python_version()}")

if system == "Windows" and machine == "ARM64":
    check("Platform", STATUS_OK, "Native ARM64 Windows Python -- QNN (NPU) execution provider can load here.")
elif system == "Windows" and emulated_hint == "ARM64":
    check(
        "Platform",
        STATUS_FAIL,
        "This is an x64 Python running under Windows-on-ARM emulation "
        f"(host is really ARM64, but this interpreter reports '{machine}'). "
        "QNNExecutionProvider will NEVER load here -- install a native "
        "ARM64 Python 3.11.x build and recreate the venv with it.",
    )
elif system == "Windows":
    check("Platform", STATUS_WARN, f"Windows on {machine} -- no Hexagon NPU here; DirectML GPU is the best case.")
else:
    check("Platform", STATUS_WARN, f"{system} -- QNN (NPU) is Windows-ARM64-only; DirectML is Windows-only too.")

# --------------------------------------------------------------------------
# 2. External system binaries this backend shells out to / links against.
# --------------------------------------------------------------------------
for binary, used_for, required in [
    ("tesseract", "OCR (Study Vision / Document Analysis text extraction)", True),
    ("pdftoppm", "PDF page rendering (poppler-utils, Document Analysis)", True),
    ("ffmpeg", "decoding browser-recorded audio (Voice Assistant)", True),
    ("espeak-ng", "pyttsx3's native voice on Linux (not needed on Windows)", system == "Linux"),
]:
    found = shutil.which(binary) is not None
    if found:
        check("System binaries", STATUS_OK, f"'{binary}' found on PATH ({used_for}).")
    elif required:
        check("System binaries", STATUS_FAIL, f"'{binary}' NOT found on PATH -- needed for {used_for}.")
    else:
        check("System binaries", STATUS_WARN, f"'{binary}' not found (only needed on Linux) -- {used_for}.")

# --------------------------------------------------------------------------
# 3. Python packages -- import-only checks, no model loading yet.
# --------------------------------------------------------------------------
core_packages = ["fastapi", "uvicorn", "onnxruntime", "cv2", "PIL", "pytesseract", "pypdf", "pydub"]
for pkg in core_packages:
    if _module_importable(pkg):
        check("Python packages", STATUS_OK, f"'{pkg}' importable.")
    else:
        check("Python packages", STATUS_FAIL, f"'{pkg}' NOT importable -- pip install -r requirements.txt")

optional_packages = {
    "onnxruntime_genai": "local LLM reasoning (Study Vision 'ask AI' / Voice Assistant)",
    "speech_recognition": "PocketSphinx STT fallback",
    "pyttsx3": "CPU TTS fallback",
    "transformers": "ONNX Whisper STT (NPU/GPU-capable path)",
    "piper_phonemize": "ONNX Piper TTS (NPU/GPU-capable path)",
}
for pkg, used_for in optional_packages.items():
    if _module_importable(pkg):
        check("Python packages (optional)", STATUS_OK, f"'{pkg}' importable ({used_for}).")
    else:
        check("Python packages (optional)", STATUS_WARN, f"'{pkg}' not installed -- {used_for} won't be available.")

# --------------------------------------------------------------------------
# 4. Execution providers actually registered with ONNX Runtime.
# --------------------------------------------------------------------------
try:
    import onnxruntime as ort

    available_eps = set(ort.get_available_providers())
    for ep in ("QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
        if ep in available_eps:
            check("ONNX Runtime providers", STATUS_OK, f"{ep} registered.")
        elif ep == "CPUExecutionProvider":
            check("ONNX Runtime providers", STATUS_FAIL, f"{ep} NOT registered -- onnxruntime install is broken.")
        else:
            check("ONNX Runtime providers", STATUS_WARN, f"{ep} not registered (expected unless installed for it).")
except ImportError:
    check("ONNX Runtime providers", STATUS_FAIL, "onnxruntime is not importable -- can't check providers at all.")

# --------------------------------------------------------------------------
# 5. Which models are actually exported on disk, per the same
#    is_available() logic the running server itself uses -- this
#    reads app/core/config.py's directories and each backend's own
#    static availability check, exactly as app/api/settings.py does.
# --------------------------------------------------------------------------
try:
    from app.core.config import LLM_EP_PRIORITY, LLM_MODELS_DIR, STT_MODEL_DIR, TTS_MODEL_DIR

    from app.models.llm.onnx_genai_backend import OnnxGenAILLM

    any_llm = False
    for subdir, ep_label in LLM_EP_PRIORITY:
        if OnnxGenAILLM.is_available(LLM_MODELS_DIR / subdir):
            check("Exported models", STATUS_OK, f"LLM ({ep_label}) exported at model_weights/llm/{subdir}/.")
            any_llm = True
    if not any_llm:
        check(
            "Exported models",
            STATUS_WARN,
            "No LLM model exported yet -- Study Vision/Voice will run OCR/STT only, "
            "no AI answer. Run scripts/export_llm_model.py.",
        )

    from app.models.stt.whisper_onnx import OnnxWhisperSTT
    from app.models.stt.pocketsphinx_stt import PocketSphinxSTT

    if OnnxWhisperSTT.is_available(STT_MODEL_DIR):
        check("Exported models", STATUS_OK, "Whisper ONNX STT model exported.")
    elif PocketSphinxSTT.is_available():
        check("Exported models", STATUS_WARN, "No Whisper ONNX model -- STT will use the PocketSphinx CPU fallback.")
    else:
        check("Exported models", STATUS_FAIL, "No STT backend available at all (neither Whisper ONNX nor PocketSphinx).")

    from app.models.tts.piper_onnx_tts import PiperOnnxTTS
    from app.models.tts.pyttsx3_tts import Pyttsx3TTS

    if PiperOnnxTTS.is_available(TTS_MODEL_DIR):
        check("Exported models", STATUS_OK, "Piper ONNX TTS model exported.")
    elif Pyttsx3TTS.is_available():
        check("Exported models", STATUS_WARN, "No Piper ONNX model -- TTS will use the pyttsx3 CPU fallback.")
    else:
        check("Exported models", STATUS_FAIL, "No TTS backend available at all (neither Piper ONNX nor pyttsx3).")
except Exception as exc:  # noqa: BLE001 -- report and keep going rather than crash the whole preflight
    check("Exported models", STATUS_FAIL, f"Could not check exported models -- {exc}")

# --------------------------------------------------------------------------
# 6. OCR model dir + frontend production build.
# --------------------------------------------------------------------------
ocr_dir = BACKEND_DIR / "model_weights" / "ocr"
if any(ocr_dir.glob("*")) if ocr_dir.exists() else False:
    check("Exported models", STATUS_OK, "OCR model files present in model_weights/ocr/.")
else:
    check(
        "Exported models",
        STATUS_WARN,
        "model_weights/ocr/ looks empty -- OCR falls back to the pytesseract CLI wrapper, which is fine "
        "(that's the default path, not an error), just slower than an exported ONNX detector.",
    )

frontend_dist = BACKEND_DIR.parent / "frontend" / "dist"
if (frontend_dist / "index.html").exists():
    check("Frontend build", STATUS_OK, f"Production build found at {frontend_dist} -- backend will serve the UI directly.")
else:
    check(
        "Frontend build",
        STATUS_WARN,
        f"No production build at {frontend_dist} -- run `npm run build` in frontend/ before demoing, "
        "or run `npm run dev` separately for local development.",
    )

# --------------------------------------------------------------------------
# 7. Disk space where models/history/settings live.
# --------------------------------------------------------------------------
try:
    free_gb = shutil.disk_usage(BACKEND_DIR).free / (1024**3)
    if free_gb < 1:
        check("Disk space", STATUS_FAIL, f"Only {free_gb:.1f} GB free near {BACKEND_DIR} -- too little for model exports.")
    elif free_gb < 5:
        check("Disk space", STATUS_WARN, f"{free_gb:.1f} GB free near {BACKEND_DIR} -- fine for now, tight if you export more models.")
    else:
        check("Disk space", STATUS_OK, f"{free_gb:.1f} GB free near {BACKEND_DIR}.")
except OSError as exc:
    check("Disk space", STATUS_WARN, f"Could not check disk space -- {exc}")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def main() -> int:
    sections: dict[str, list[tuple[str, str]]] = {}
    for section, status, message in results:
        sections.setdefault(section, []).append((status, message))

    print("=" * 78)
    print(" SnapSight AI -- deployment preflight check")
    print("=" * 78)
    for section, entries in sections.items():
        print(f"\n{section}:")
        for status, message in entries:
            print(f"  {_ICON[status]} {message}")

    worst = STATUS_OK
    counts = {STATUS_OK: 0, STATUS_WARN: 0, STATUS_FAIL: 0}
    for _, status, _msg in results:
        counts[status] += 1
        if status == STATUS_FAIL:
            worst = STATUS_FAIL
        elif status == STATUS_WARN and worst != STATUS_FAIL:
            worst = STATUS_WARN

    print("\n" + "=" * 78)
    print(f" {counts[STATUS_OK]} OK, {counts[STATUS_WARN]} warning(s), {counts[STATUS_FAIL]} failure(s)")
    if worst == STATUS_OK:
        print(" Ready to demo -- every checked path has at least a CPU fallback, and no blocking issues found.")
    elif worst == STATUS_WARN:
        print(" Ready to demo, with fallbacks in play (see WARN lines above) -- nothing here is faked, just slower/CPU-only.")
    else:
        print(" NOT ready -- fix the FAIL line(s) above before demoing.")
    print("=" * 78)

    return {STATUS_OK: 0, STATUS_WARN: 1, STATUS_FAIL: 2}[worst]


if __name__ == "__main__":
    sys.exit(main())
