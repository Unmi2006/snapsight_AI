"""
Gets a local LLM model folder in place under model_weights/llm/ so Study
Vision Mode's "ask AI" step activates instead of reporting itself
unavailable.

THIS MUST BE RUN ON A MACHINE WITH INTERNET ACCESS TO HUGGING FACE.
The dev sandbox this repo was built in has a network allowlist (pypi/npm/
github/crates only, no huggingface.co) and cannot fetch the ~2-3GB model
weights itself -- that's why this can't just already be sitting in the
zip. Run this script on your own laptop / the target machine instead.

WHICH MODEL: Phi-3.5-mini-instruct-onnx (Microsoft, ~3.8B params, int4
quantized). Chosen because it's the one model in this class Microsoft
ships as ready-made ONNX Runtime GenAI exports for CPU, DirectML (GPU),
AND -- on recent releases -- QNN (Snapdragon NPU), so the SAME backend
code (app/models/llm/onnx_genai_backend.py) works across all three
without touching app code, exactly like PP-OCRv4 does for OCR.

--------------------------------------------------------------------------
USAGE (one command each)
--------------------------------------------------------------------------
    pip install huggingface_hub

    python scripts/export_llm_model.py --fetch cpu     # safety-net, always do this one
    python scripts/export_llm_model.py --fetch dml     # Windows DirectML GPU
    python scripts/export_llm_model.py --fetch qnn     # Snapdragon NPU, if a QNN export exists for this release
    python scripts/export_llm_model.py --fetch cpu dml # fetch more than one in one run

Each --fetch downloads straight into model_weights/llm/<subdir>/ (no manual
copying) and then installs the matching onnxruntime-genai package for you
by shelling out to pip -- EXCEPT it will refuse to install two conflicting
EPs (onnxruntime-genai vs onnxruntime-genai-directml share an import name;
see the conflict check below) so it won't silently break your env.

Then verify what's actually loadable:
    python scripts/export_llm_model.py --verify

Once a folder with genai_config.json exists under model_weights/llm/, just
restart the backend -- LLMService picks it up automatically (NPU > GPU >
CPU priority, see app/core/config.py:LLM_EP_PRIORITY) and Study Vision's
"ask AI" step switches on. No app code changes needed.

--------------------------------------------------------------------------
ALTERNATIVE -- Qualcomm AI Hub models (Llama-3.2-3B-Instruct etc.)
--------------------------------------------------------------------------
Qualcomm AI Hub lists larger instruction-tuned models pre-optimized
specifically for Snapdragon X Elite/Plus NPUs via `qai_hub_models`, but
they're typically consumed through Qualcomm's own Genie runtime rather
than onnxruntime-genai -- a different integration than this script covers.
Treat that as a documented Phase 2+ follow-up; check
https://aihub.qualcomm.com for current availability before committing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
LLM_DIR = BACKEND_ROOT / "model_weights" / "llm"

# subdir -> (HF repo, allow_patterns glob, genai package to pip install, EP label)
_FETCH_SPECS = {
    "cpu": (
        "microsoft/Phi-3.5-mini-instruct-onnx",
        "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/*",
        "onnxruntime-genai>=0.4.0",
        "CPUExecutionProvider",
    ),
    "dml": (
        "microsoft/Phi-3.5-mini-instruct-onnx",
        "directml/directml-int4-awq-block-128/*",
        "onnxruntime-genai-directml>=0.4.0",
        "DmlExecutionProvider (Windows only)",
    ),
    # No fixed pattern for qnn: availability/folder-naming varies by release.
    # Handled separately in fetch_qnn() below -- lists the repo tree and
    # picks the first folder starting with 'qnn' rather than guessing a path.
    "qnn": (
        "microsoft/Phi-3.5-mini-instruct-onnx",
        None,
        None,  # decided at runtime once we know the QNN package name convention exists
        "QNNExecutionProvider (Snapdragon NPU only)",
    ),
}

# onnxruntime-genai and onnxruntime-genai-directml register the same
# `onnxruntime_genai` import name and will silently shadow each other if
# both are installed -- same rule as onnxruntime vs onnxruntime-directml.
_CONFLICTING_GENAI_PACKAGES = ["onnxruntime-genai", "onnxruntime-genai-directml", "onnxruntime-genai-cuda"]


def _pip_install(package: str) -> None:
    base_name = package.split(">=")[0].split("==")[0]
    for installed in _CONFLICTING_GENAI_PACKAGES:
        if installed != base_name:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", installed], check=False)
    print(f"  installing {package} ...")
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


def fetch_standard(subdir: str) -> None:
    repo, pattern, package, ep_label = _FETCH_SPECS[subdir]
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency. Run: pip install huggingface_hub")
        sys.exit(1)

    dest = LLM_DIR / subdir
    dest.mkdir(parents=True, exist_ok=True)
    tmp = BACKEND_ROOT / f"_tmp_llm_{subdir}"

    print(f"[{subdir}] downloading {pattern} from {repo} (this is a multi-GB download, be patient) ...")
    snapshot_download(repo, allow_patterns=pattern, local_dir=str(tmp))

    # allow_patterns preserves the repo's subfolder structure under tmp/;
    # find wherever genai_config.json actually landed and flatten it into dest.
    found = list(tmp.rglob("genai_config.json"))
    if not found:
        print(f"  ERROR: no genai_config.json found under {tmp} -- the HF repo layout may have "
              f"changed. Check https://huggingface.co/{repo}/tree/main and update the pattern above.")
        sys.exit(1)
    src_dir = found[0].parent
    for item in src_dir.iterdir():
        item.rename(dest / item.name)
    print(f"  -> model files in place at {dest}")

    _pip_install(package)
    print(f"[{subdir}] done. EP: {ep_label}")


def fetch_qnn() -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("Missing dependency. Run: pip install huggingface_hub")
        sys.exit(1)

    repo = _FETCH_SPECS["qnn"][0]
    api = HfApi()
    files = api.list_repo_files(repo)
    qnn_folders = sorted({f.split("/")[0] for f in files if f.lower().startswith("qnn")})
    if not qnn_folders:
        print(
            f"No 'qnn*' folder found in {repo} on this Hugging Face release -- QNN export "
            "availability varies over time and isn't guaranteed. This is a genuine current gap, "
            "not something to fake: DirectML GPU (--fetch dml) is real hardware acceleration on "
            "the same Snapdragon PC and is an honest fallback story for the LLM step specifically."
        )
        sys.exit(1)

    folder = qnn_folders[0]
    print(f"[qnn] found '{folder}' in {repo}, downloading ...")
    dest = LLM_DIR / "qnn"
    dest.mkdir(parents=True, exist_ok=True)
    tmp = BACKEND_ROOT / "_tmp_llm_qnn"
    snapshot_download(repo, allow_patterns=f"{folder}/*", local_dir=str(tmp))
    src_dir = tmp / folder
    for item in src_dir.iterdir():
        item.rename(dest / item.name)
    print(f"  -> model files in place at {dest}")

    print(
        "  NOTE: the matching onnxruntime-genai QNN wheel is newer and less standardized than "
        "onnxruntime-qnn (the raw ORT one OCR/vision use) -- check "
        "https://onnxruntime.ai/docs/genai/howto/install for the current QNN install command "
        "for your release before running --verify."
    )


def verify() -> None:
    found_any = False
    for subdir in ("qnn", "dml", "cpu"):
        model_dir = LLM_DIR / subdir
        config_path = model_dir / "genai_config.json"
        if not config_path.exists():
            print(f"NOT FOUND: {config_path}")
            continue

        found_any = True
        print(f"OK: {config_path}")
        try:
            import onnxruntime_genai as og

            model = og.Model(str(model_dir))
            print(f"  -> loaded successfully with onnxruntime-genai ({subdir} folder)")
            del model
        except ImportError:
            print("  -> genai_config.json present but onnxruntime-genai isn't installed "
                  "(run --fetch again, or pip install the matching package manually)")
        except Exception as exc:  # noqa: BLE001
            print(f"  -> found the folder but failed to load: {exc}")

    if not found_any:
        print("\nNo LLM model folder found under model_weights/llm/. Run --fetch cpu first.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", nargs="+", choices=["cpu", "dml", "qnn"], metavar="{cpu,dml,qnn}",
                         help="Download + install one or more EP variants, e.g. --fetch cpu dml")
    parser.add_argument("--verify", action="store_true", help="Check what's currently loadable")
    args = parser.parse_args()

    if not args.fetch and not args.verify:
        parser.print_help()
        return

    if args.fetch:
        for subdir in args.fetch:
            if subdir == "qnn":
                fetch_qnn()
            else:
                fetch_standard(subdir)

    if args.verify or args.fetch:
        print("\n--- verify ---")
        verify()


if __name__ == "__main__":
    main()
