"""
Gets model_weights/ocr/rec.onnx in place so PaddleOCRONNX (the NPU-capable
OCR backend) activates instead of the Tesseract fallback.

This sandbox's network policy can't reach the hosts these models live on,
so this script can't be run here -- run it on your own machine (the x64
dev box is fine; you don't need the Snapdragon PC for this step).

TWO OPTIONS, pick one:

--------------------------------------------------------------------------
OPTION A (fastest -- pre-converted ONNX, no PaddlePaddle install needed)
--------------------------------------------------------------------------
Download the already-converted PP-OCRv4 English recognition model from
Hugging Face (SWHL/RapidOCR mirrors + xberg-io/paddleocr-onnx-models both
host this):

    pip install huggingface_hub
    python -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id='xberg-io/paddleocr-onnx-models', filename='en_PP-OCRv4_rec_infer.onnx')
shutil.copy(p, 'model_weights/ocr/rec.onnx')
print('Saved to model_weights/ocr/rec.onnx')
"

Then run this file with --verify to sanity-check the shape:
    python scripts/export_ocr_model.py --verify

--------------------------------------------------------------------------
OPTION B (manual -- official PaddleOCR + paddle2onnx conversion)
--------------------------------------------------------------------------
1. Install tooling:
     pip install paddlepaddle paddle2onnx onnxruntime

2. Download the official PP-OCRv4 English recognition inference model:
     wget -P ./tmp_ocr https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar
     cd tmp_ocr && tar xf en_PP-OCRv4_mobile_rec_infer.tar && cd ..

3. Convert to ONNX with a fixed input shape (batch=1, 3 channels, 48 tall,
   320 wide -- matches REC_INPUT_HEIGHT in paddle_ocr_onnx.py):
     paddle2onnx \
       --model_dir ./tmp_ocr/en_PP-OCRv4_mobile_rec_infer \
       --model_filename inference.pdmodel \
       --params_filename inference.pdiparams \
       --save_file ./model_weights/ocr/rec.onnx \
       --opset_version 14 \
       --enable_dev_version True

     python -m paddle2onnx.optimize \
       --input_model ./model_weights/ocr/rec.onnx \
       --output_model ./model_weights/ocr/rec.onnx \
       --input_shape_dict "{'x':[1,3,48,320]}"

--------------------------------------------------------------------------
Either way, once model_weights/ocr/rec.onnx exists, restart the backend --
OCRService will detect it automatically and switch from Tesseract to the
ONNX path (and pick up NPU acceleration once you're on the Snapdragon
device with onnxruntime-qnn installed).

For the DETECTION model too (better than this repo's simplified OpenCV
region proposal), the equivalent PP-OCRv4 detector is
`PP-OCRv4_mobile_det_infer.tar` from the same bj.bcebos.com host --
convert it the same way as Option B above and wire it into
paddle_ocr_onnx.py's `_propose_text_regions` as a follow-up.
"""
import sys
from pathlib import Path

REC_PATH = Path(__file__).resolve().parent.parent / "model_weights" / "ocr" / "rec.onnx"


def verify():
    if not REC_PATH.exists():
        print(f"NOT FOUND: {REC_PATH}")
        print("Follow Option A or B in this file's docstring first.")
        sys.exit(1)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(REC_PATH), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"OK: {REC_PATH}")
    print(f"Input name: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
    print(f"Output(s): {[o.name for o in sess.get_outputs()]}")
    print(
        "Expected input shape roughly [1, 3, 48, W] (W flexible/dynamic is fine). "
        "If height isn't 48, update REC_INPUT_HEIGHT in app/models/ocr/paddle_ocr_onnx.py."
    )


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        print(__doc__)
