"""
Gets a Piper ONNX voice in place under model_weights/tts/ so Voice
Assistant's TTS step uses the NPU/GPU-capable PiperOnnxTTS backend
instead of the pyttsx3 CPU fallback.

This sandbox's network policy can't reach Hugging Face, so this script
can't fetch anything here -- run it on your own machine, same as the
other export_*.py scripts.

WHICH MODEL: any Piper voice from rhasspy/piper-voices, e.g.
en_US-lessac-medium (small, fast, good English quality).

--------------------------------------------------------------------------
STEP 1 -- download a voice
--------------------------------------------------------------------------
    pip install huggingface_hub
    python -c "
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('model_weights/tts', exist_ok=True)
repo = 'rhasspy/piper-voices'
for fname in [
    'en/en_US/lessac/medium/en_US-lessac-medium.onnx',
    'en/en_US/lessac/medium/en_US-lessac-medium.onnx.json',
]:
    p = hf_hub_download(repo_id=repo, filename=fname)
    shutil.copy(p, f'model_weights/tts/{fname.rsplit(chr(47), 1)[-1]}')
    print('saved', fname)
"

--------------------------------------------------------------------------
STEP 2 -- install the phonemizer (bundles/needs espeak-ng)
--------------------------------------------------------------------------
    pip install piper-phonemize
    # If wheels aren't available for your platform/arch, also install the
    # native espeak-ng package:
    #   sudo apt install espeak-ng     (Linux)
    #   brew install espeak-ng         (macOS)

--------------------------------------------------------------------------
STEP 3 -- verify
--------------------------------------------------------------------------
    python scripts/export_tts_model.py --verify

--------------------------------------------------------------------------
NPU (Snapdragon QNN) note
--------------------------------------------------------------------------
Qualcomm AI Hub does not currently list a first-party Piper/VITS TTS
export for the Hexagon NPU as of this writing -- check
https://aihub.qualcomm.com yourself before the competition and update
this note. If none exists, DirectML GPU or CPU is the honest story for
the TTS step specifically -- same rule as everywhere else in this repo
about not claiming NPU without verifying `active_provider`.

Either way, once a <voice>.onnx + <voice>.onnx.json pair exists under
model_weights/tts/, restart the backend -- TTSService detects it
automatically and switches from pyttsx3 to the ONNX path.
"""
import sys
from pathlib import Path

TTS_DIR = Path(__file__).resolve().parent.parent / "model_weights" / "tts"


def verify():
    onnx_files = sorted(TTS_DIR.glob("*.onnx")) if TTS_DIR.exists() else []
    if not onnx_files:
        print(f"NOT FOUND: no .onnx file in {TTS_DIR}")
        print("Follow STEP 1 in this file's docstring first.")
        sys.exit(1)

    voice_path = onnx_files[0]
    config_path = Path(str(voice_path) + ".json")
    if not config_path.exists():
        print(f"NOT FOUND: {config_path} (needed alongside the .onnx file)")
        sys.exit(1)

    import json

    import onnxruntime as ort

    sess = ort.InferenceSession(str(voice_path), providers=["CPUExecutionProvider"])
    print(f"OK: {voice_path}")
    print(f"  Inputs:  {[(i.name, i.shape) for i in sess.get_inputs()]}")
    print(f"  Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"  Sample rate (from config): {cfg.get('audio', {}).get('sample_rate')}")
    print(f"  espeak voice (from config): {cfg.get('espeak', {}).get('voice')}")

    try:
        import piper_phonemize  # noqa: F401

        print("  piper-phonemize importable: yes")
    except ImportError:
        print("  piper-phonemize importable: NO -- pip install piper-phonemize (STEP 2)")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        print(__doc__)
