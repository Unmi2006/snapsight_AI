"""
Gets a Whisper ONNX model folder in place under model_weights/stt/ so
Voice Assistant's STT step uses the NPU/GPU-capable OnnxWhisperSTT
backend instead of the PocketSphinx CPU fallback.

This sandbox's network policy can't reach Hugging Face, so this script
can't fetch anything here -- run it on your own machine, same as
export_ocr_model.py / export_llm_model.py.

WHICH MODEL: whisper-tiny.en (39M params, English-only) by default --
small enough to be genuinely fast on CPU/GPU/NPU and to keep the
no-KV-cache decode loop in app/models/stt/whisper_onnx.py (see that
file's docstring) cheap. Swap in whisper-base.en for a bit more accuracy
if the demo device can afford the extra latency.

--------------------------------------------------------------------------
STEP 1 -- export via Hugging Face Optimum
--------------------------------------------------------------------------
    pip install optimum[exporters] onnxruntime

    optimum-cli export onnx \
      --model openai/whisper-tiny.en \
      --task automatic-speech-recognition \
      ./tmp_stt

    mkdir -p model_weights/stt
    cp tmp_stt/encoder_model.onnx model_weights/stt/
    cp tmp_stt/decoder_model.onnx model_weights/stt/
    cp tmp_stt/tokenizer.json model_weights/stt/ 2>/dev/null || true
    cp tmp_stt/vocab.json model_weights/stt/ 2>/dev/null || true
    cp tmp_stt/merges.txt model_weights/stt/ 2>/dev/null || true
    cp tmp_stt/tokenizer_config.json model_weights/stt/ 2>/dev/null || true
    cp tmp_stt/special_tokens_map.json model_weights/stt/ 2>/dev/null || true
    cp tmp_stt/normalizer.json model_weights/stt/ 2>/dev/null || true
    cp tmp_stt/generation_config.json model_weights/stt/ 2>/dev/null || true
    cp tmp_stt/added_tokens.json model_weights/stt/ 2>/dev/null || true

Note: optimum also emits decoder_with_past_model.onnx and/or
decoder_model_merged.onnx -- those are for the KV-cache decode path this
MVP intentionally doesn't implement yet (see the docstring in
app/models/stt/whisper_onnx.py). Copying them over is harmless, they're
just unused; wiring them in is the documented follow-up optimization.

--------------------------------------------------------------------------
STEP 2 -- install the extra Python dependencies the ONNX path needs
--------------------------------------------------------------------------
    pip install transformers librosa soundfile
(these are commented out in requirements.txt so the base app doesn't
require them just to boot -- uncomment once you're using this path)

--------------------------------------------------------------------------
STEP 3 -- verify
--------------------------------------------------------------------------
    python scripts/export_stt_model.py --verify

--------------------------------------------------------------------------
NPU (Snapdragon QNN) note
--------------------------------------------------------------------------
Qualcomm AI Hub lists pre-optimized Whisper encoder/decoder models tuned
for the Hexagon NPU (search "Whisper" at https://aihub.qualcomm.com).
Those ship as separate encoder/decoder files with their own I/O contract,
which may not exactly match optimum's export names used above ("input_ids",
"encoder_hidden_states", "input_features") -- check the model card's
input/output signature before assuming a drop-in swap. Same rule as
everywhere else in this repo: verify the active_provider InferenceEngine
actually reports before claiming NPU acceleration for the demo, don't
assume it from the download source.

Either way, once model_weights/stt/encoder_model.onnx and
decoder_model.onnx exist, restart the backend -- STTService detects them
automatically and switches from PocketSphinx to the ONNX path.
"""
import sys
from pathlib import Path

STT_DIR = Path(__file__).resolve().parent.parent / "model_weights" / "stt"


def verify():
    encoder = STT_DIR / "encoder_model.onnx"
    decoder = STT_DIR / "decoder_model.onnx"
    if not encoder.exists() or not decoder.exists():
        print(f"NOT FOUND: {encoder} and/or {decoder}")
        print("Follow STEP 1 in this file's docstring first.")
        sys.exit(1)

    import onnxruntime as ort

    for path in (encoder, decoder):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        print(f"OK: {path}")
        print(f"  Inputs:  {[(i.name, i.shape) for i in sess.get_inputs()]}")
        print(f"  Outputs: {[(o.name, o.shape) for o in sess.get_outputs()]}")

    tokenizer_ok = (STT_DIR / "tokenizer.json").exists() or (STT_DIR / "vocab.json").exists()
    print(f"Tokenizer files present: {tokenizer_ok}")
    if not tokenizer_ok:
        print("WARNING: no tokenizer files found next to the .onnx files -- copy them from the export too.")

    try:
        import transformers  # noqa: F401
        import librosa  # noqa: F401

        print("transformers + librosa importable: yes")
    except ImportError as exc:
        print(f"transformers/librosa NOT importable ({exc}) -- pip install transformers librosa soundfile")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        print(__doc__)
