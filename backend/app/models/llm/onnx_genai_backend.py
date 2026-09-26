"""
LLM backend built on onnxruntime-genai (`import onnxruntime_genai as og`) --
the text-generation equivalent of what InferenceEngine does for OCR/vision:
ONNX Runtime, NPU-first, honest about what actually ran.

HONESTY NOTE -- read this before trusting `active_provider` on the result:
Unlike raw `onnxruntime.InferenceSession` (app/core/inference_engine.py),
onnxruntime-genai does NOT expose a `session.get_providers()` you can
inspect afterwards to independently prove which execution provider a
loaded model actually ran on. The EP is baked in at export time: Microsoft
ships Phi-3.5-mini-instruct-onnx as separate folders per EP (e.g.
`cpu-int4-rtn-block-32-acc-level-4/`, `directml-int4-awq-block-128/`,
a `qnn-int4/` variant on some releases), each with its own
`genai_config.json` declaring which provider it was built for.

So "active_provider" here is READ OUT of that config file, not
re-verified by actually probing the provider the way OCR's ONNX path is.
This class never hardcodes a provider label -- if the config is missing
or unparseable it honestly reports "unknown" rather than guessing.

Requires the matching onnxruntime-genai package to be installed for the
EP you want (onnxruntime-genai for CPU, onnxruntime-genai-directml for
DirectML GPU -- see requirements-directml.txt; QNN/NPU support is newer
and less standardized, see scripts/export_llm_model.py for the current
state). Only one of these should be installed at a time -- same
import-name-conflict rule as onnxruntime vs onnxruntime-directml.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from app.models.llm.base import GenerationConfig, LLMResult

logger = logging.getLogger("snapsight.llm.onnx_genai")

_PROVIDER_KEY_TO_LABEL = {
    "dml": "DmlExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def _declared_provider(model_dir: Path) -> str:
    """Reads genai_config.json's provider_options to see what EP this
    specific exported model folder declares -- never guessed, never
    inferred from the folder name alone."""
    config_path = model_dir / "genai_config.json"
    if not config_path.exists():
        return "unknown"
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        provider_options = (
            cfg.get("model", {}).get("decoder", {}).get("session_options", {}).get("provider_options", [])
        )
        if not provider_options:
            # No explicit EP block in the config means genai's own CPU
            # default is what will run -- not a guess, that's documented
            # onnxruntime-genai behavior.
            return "CPUExecutionProvider"
        key = next(iter(provider_options[0].keys()), "cpu")
        return _PROVIDER_KEY_TO_LABEL.get(key, key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse %s: %s", config_path, exc)
        return "unknown"


class OnnxGenAILLM:
    name = "onnx_genai"

    def __init__(self, model_dir: str | Path, model_display_name: str | None = None):
        import onnxruntime_genai as og  # optional dependency, imported lazily

        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"LLM model folder not found: {self.model_dir}")

        self.model_display_name = model_display_name or self.model_dir.name
        self.declared_provider = _declared_provider(self.model_dir)

        t0 = time.perf_counter()
        self._model = og.Model(str(self.model_dir))
        self._tokenizer = og.Tokenizer(self._model)
        self.load_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        self._og = og

        logger.info(
            "LLM loaded: %s from %s (declared provider: %s, load %.1fms)",
            self.model_display_name, self.model_dir, self.declared_provider, self.load_time_ms,
        )

    @staticmethod
    def is_available(model_dir: str | Path) -> bool:
        model_dir = Path(model_dir)
        if not (model_dir / "genai_config.json").exists():
            return False
        try:
            import onnxruntime_genai  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> LLMResult:
        config = config or GenerationConfig()
        og = self._og

        t0 = time.perf_counter()
        input_tokens = self._tokenizer.encode(prompt)

        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=len(input_tokens) + config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
        )

        generator = og.Generator(self._model, params)
        # append_tokens() is the current (>=0.4) onnxruntime-genai API for
        # seeding the prompt; older releases used `params.input_ids = ...`
        # instead. Pin the version in requirements.txt so this stays correct.
        generator.append_tokens(input_tokens)

        output_tokens: list[int] = []
        while not generator.is_done():
            generator.generate_next_token()
            output_tokens.append(generator.get_next_tokens()[0])

        latency_ms = (time.perf_counter() - t0) * 1000
        text = self._tokenizer.decode(output_tokens)

        return LLMResult(
            text=text.strip(),
            engine=self.name,
            model_name=self.model_display_name,
            active_provider=self.declared_provider,
            latency_ms=round(latency_ms, 3),
            prompt_tokens=len(input_tokens),
            completion_tokens=len(output_tokens),
        )
