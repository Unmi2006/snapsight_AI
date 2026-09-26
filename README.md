# SnapSight AI — Phases 0 + 1 + 2 + 3 + 4 + 5

Privacy-preserving on-device multimodal AI assistant for the Snapdragon AI Lab
Build & Present Challenge. Phase 0 shipped the hardware verification layer;
Phase 1 adds camera capture, OCR, and document ingestion on top of it. No
fake demo, no hardcoded EP claims, no hardcoded OCR text.

## Phase 1: what's new

- **OCR module** (`app/models/ocr/`): `TesseractOCR` (real, CPU, works today
  with zero downloads) and `PaddleOCRONNX` (routes through `InferenceEngine`
  for NPU/GPU acceleration once you export `rec.onnx` — see
  `scripts/export_ocr_model.py`). `OCRService` picks whichever is available
  and always reports which one ran and on what execution provider.
- **`/api/camera/capture`** — accepts one captured frame, runs OCR, returns
  text + bounding boxes.
- **`/api/documents/analyze`** — accepts a PDF or image. Digital (born-text)
  PDFs get instant text extraction with no OCR; scanned PDFs and images are
  rasterized and run through the same OCR pipeline, page by page.
- **Camera Mode page** — real `getUserMedia` webcam preview, capture button,
  bounding-box overlay on a canvas.
- **Document Analysis page** — file upload, shows extracted text plus which
  OCR engine/provider/latency was used per page.

Both new endpoints were tested end-to-end in development (synthetic image →
Tesseract → correct text back; a digital PDF → instant extraction, no OCR;
a scanned image → OCR path; an unsupported file → clean 400 error).

### System dependencies for OCR (install these, not just pip packages)

```bash
# Linux
sudo apt install tesseract-ocr poppler-utils
# macOS
brew install tesseract poppler
# Windows: Tesseract installer from https://github.com/UB-Mannheim/tesseract/wiki
#          poppler from https://github.com/oschwartz10612/poppler-windows/releases
#          (add both install dirs to PATH)
```

Without these, `pip install -r requirements.txt` succeeds but OCR calls
fail at runtime with a clear "binary not found" error, not a silent fake
response.

### Getting real NPU/GPU-accelerated OCR instead of the Tesseract fallback

Run `python scripts/export_ocr_model.py` (prints instructions — network in
this dev sandbox can't reach the model hosts, so this step happens on your
own machine) to get `model_weights/ocr/rec.onnx` in place, then restart the
backend. `OCRService` will detect it automatically and switch backends.
`--verify` checks the exported model's input shape.

## Phase 2: what's new

- **LLM module** (`app/models/llm/`): `OnnxGenAILLM`, built on
  `onnxruntime-genai`, the GenAI counterpart to `InferenceEngine` for text
  generation. Reads the execution provider straight out of each exported
  model folder's `genai_config.json` rather than guessing — see the
  honesty note at the top of `onnx_genai_backend.py` for exactly what that
  does and doesn't prove.
- **`LLMService`** (`llm_service.py`): tries `model_weights/llm/qnn` →
  `dml` → `cpu` in that order (NPU → GPU → CPU, same priority as OCR/vision)
  and picks the first folder that exists and loads. If none exist, it
  **does not crash the app** — it marks itself unavailable and every
  endpoint that needs it reports that plainly instead of faking a reply.
- **Multimodal orchestrator** (`app/orchestrator/study_vision.py`): turns
  OCR output + a task the user picked into one LLM prompt. Six modes match
  the brief's example questions directly: Explain this / Summarize this
  page / Explain simply (2nd-year engineering level) / Find the bug /
  Extract the equation / Ask a custom question. This module only builds
  prompt strings — it never generates or hardcodes an answer itself.
- **`/api/vision/study`** — the Study Vision endpoint: runs OCR on the
  captured frame, builds the prompt via the orchestrator, and (if a model
  is loaded) reasons over it with the local LLM. Returns OCR results even
  when the LLM step is unavailable, so the page degrades gracefully rather
  than failing outright.
- **`/api/vision/modes`** and **`/api/vision/llm-status`** — drive the
  Study Vision page's mode buttons and the "LLM not installed yet" banner;
  the frontend never hardcodes the mode list, it asks the backend.
- **Study Vision Mode page** — camera capture (reusing the same
  `getUserMedia` pattern as Camera Mode) or file upload, mode buttons,
  optional content-type hint, a custom-question box, and a results view
  showing the OCR text plus the AI answer with engine/provider/latency
  badges for both steps.
- **Robustness fix carried over from the Tesseract PATH issue**:
  `OCRService` no longer crashes the whole backend if Tesseract isn't
  installed — it reports itself unavailable and camera/document/vision
  endpoints return a clean 503 instead of the server failing to boot.

Tested end-to-end in development: a synthetic code snippet image →
Tesseract OCR → orchestrator builds a `find_bug` prompt → LLM correctly
reports itself unavailable (no model installed in this dev sandbox,
network here can't reach Hugging Face) rather than returning a fake
answer. Same for the `custom` mode's "requires a question" validation and
an unknown-mode 400.

### Getting a working local LLM

`onnxruntime-genai` is in `requirements.txt` (CPU) /
`requirements-directml.txt` (GPU), but the model weights themselves aren't
shipped in this repo — same reasoning as `rec.onnx` in Phase 1: this dev
sandbox's network can't reach Hugging Face. Run
`python scripts/export_llm_model.py` (prints full instructions) on a
machine with internet access to fetch Microsoft's Phi-3.5-mini-instruct
ONNX GenAI export (CPU and/or DirectML variant), drop it into
`model_weights/llm/cpu/` or `model_weights/llm/dml/`, then restart the
backend. `--verify` checks the exported folder loads correctly.

Without a model in place, Study Vision Mode still runs OCR end-to-end —
only the "AI answer" section shows as unavailable, with a link to the
export script, exactly like the Tesseract/rec.onnx pattern from Phase 1.

## Phase 3: what's new (Voice Assistant)

- **STT module** (`app/models/stt/`): `PocketSphinxSTT` (real, CPU, works
  today with zero downloads -- PocketSphinx's default English model ships
  inside the pip wheel itself, same "always-available fallback" role
  Tesseract plays for OCR) and `OnnxWhisperSTT` (routes through the same
  `InferenceEngine` for NPU/GPU/CPU acceleration once you export a
  Whisper ONNX model -- see `scripts/export_stt_model.py`). `STTService`
  picks whichever is available and always reports which one ran and on
  what execution provider. `whisper_onnx.py`'s docstring documents two
  deliberate simplifications (no KV-cache decode loop; librosa's
  standard mel filterbank instead of Whisper's exact shipped one) rather
  than hiding them.
- **TTS module** (`app/models/tts/`): `Pyttsx3TTS` (real, CPU, wraps
  whatever native offline speech engine the OS already has -- espeak-ng
  on Linux, SAPI5 on Windows, NSSpeechSynthesizer on macOS -- zero model
  downloads) and `PiperOnnxTTS` (NPU/GPU/CPU-capable neural TTS via
  `InferenceEngine`, needs a Piper voice export -- see
  `scripts/export_tts_model.py`). `TTSService` mirrors the same
  selection/fallback pattern.
- **Voice orchestrator** (`app/orchestrator/voice_assistant.py`): builds
  the prompt for a spoken question. Reuses Study Vision's orchestrator
  (Phase 2) when an image is attached, so a spoken question can reference
  on-screen content with the same OCR-grounding rules; falls back to a
  plain spoken-answer-friendly system prompt otherwise. This module only
  builds prompt strings -- it never generates or hardcodes an answer.
- **`/api/voice/status`**, **`/api/voice/transcribe`**,
  **`/api/voice/speak`**, **`/api/voice/ask`** -- the full loop: record a
  question, optionally attach an image, transcribe, reason with the
  local LLM (Phase 2's `LLMService`, unchanged), and speak the answer
  back. Every stage reports its own engine/provider/latency and degrades
  independently and honestly -- e.g. if the LLM isn't installed, the
  transcript (and OCR, if an image was sent) are still returned instead
  of the whole request failing.
- **Voice Assistant page** (`frontend/src/pages/VoiceAssistant.jsx`):
  `MediaRecorder`-based mic capture, the same camera/upload pattern as
  Study Vision for optional visual context, and a results view showing
  the transcript, OCR text (if any), the AI answer, and a playable
  `<audio>` element for the spoken reply -- each section shows its own
  engine/provider/latency badges.
- **Real perf tuning in `InferenceEngine`**: every session now requests
  full graph optimization and sizes the CPU intra-op thread pool to the
  machine's logical core count. This is a genuine, measured speed change
  to the CPU fallback path (which is where most models land until
  NPU/GPU is verified present) -- not a claim about which execution
  provider is used.

Tested end-to-end in this dev sandbox (Linux, no GPU/NPU, no Whisper/Piper
models exported): with `espeak-ng` + `ffmpeg` + `pocketsphinx` installed,
`/api/voice/speak` produced a real playable WAV via pyttsx3, that same WAV
was fed back into `/api/voice/transcribe` and PocketSphinx returned real
(if imperfect, as documented) recognized text, and `/api/voice/ask` -- both
with and without an attached image -- correctly transcribed, ran OCR when
an image was present, and reported the LLM step as unavailable (no model
exported in this sandbox, same Phase 2 situation) rather than faking an
answer. Phase 1/2 endpoints (`/api/camera/capture`, `/api/hardware/info`,
`/api/hardware/verify`, `/api/benchmark/summary`) were re-checked after
the `InferenceEngine` change and still pass.

### Getting real NPU/GPU-accelerated voice instead of the CPU fallbacks

Run `python scripts/export_stt_model.py` and
`python scripts/export_tts_model.py` (both print full instructions --
network in this dev sandbox can't reach the model hosts, so this step
happens on your own machine) to get a Whisper ONNX export in
`model_weights/stt/` and a Piper voice in `model_weights/tts/`, then
restart the backend. Both services detect their ONNX model automatically
and switch backends; `--verify` on each script checks the exported files.

### System dependencies for voice (install these, not just pip packages)

```bash
# Linux
sudo apt install espeak-ng ffmpeg
# for PocketSphinx's build step, also: sudo apt install swig
# macOS
brew install espeak-ng ffmpeg
# Windows: espeak-ng installer from https://github.com/espeak-ng/espeak-ng/releases
#          ffmpeg build from https://www.gyan.dev/ffmpeg/builds/ (add to PATH)
```

Without `ffmpeg`, audio decoding fails with a clear "ffmpeg not found"
error (`app/utils/audio_loader.py`), not a silent fake transcript.
Without `espeak-ng` (Linux) or a working native TTS voice, `pyttsx3`
reports itself unavailable at startup instead of crashing the backend --
same robustness pattern as the Tesseract/PocketSphinx fallbacks.

## What's actually working right now

- FastAPI backend with `/api/hardware/info` and `/api/hardware/verify` —
  reports which ONNX Runtime execution providers (NPU/GPU/CPU) are really
  installed and runs a live micro-benchmark to prove which one actually
  executed a model, not just which one you asked for.
- `InferenceEngine` (`app/core/inference_engine.py`) — the class every
  future model (OCR, vision, Whisper, LLM, TTS) will load through. It walks
  NPU → GPU → CPU in priority order and only reports success for a provider
  it verified.
- Benchmark profiler (`app/benchmarking/profiler.py`) — a context manager
  that records latency/memory/CPU into a local SQLite file
  (`backend/model_weights/benchmarks.db`) for every inference call.
- React frontend shell with routing, a Hardware Info page (live, wired to
  the backend), a Performance Dashboard page (live, polls every 5s),
  Camera Mode, Study Vision, Document Analysis, and Voice Assistant
  (all live), plus a History page (browse/replay/delete past turns) and
  a Settings page (provider overrides, voice, defaults) — also live as
  of Phase 4.
- Single-process production serving, a deployment preflight script, and
  Windows setup/launch scripts — Phase 5, see below.

## What's NOT built yet (by design — later phases)

Nothing from the original roadmap — Phases 0 through 5 are all done.
Actually running on the Snapdragon PC itself, and validating the
NPU/QNN and DirectML paths on real Snapdragon silicon, are still
outstanding since this has only been built and tested on a regular x64
Linux dev machine so far — `scripts/check_deployment.py` (Phase 5) is
built specifically to surface that gap clearly instead of hiding it.

## Backend setup

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

On your x64 dev machine, once you want GPU acceleration via DirectML
(your chosen first fallback priority):

```bash
pip uninstall onnxruntime -y
pip install -r requirements-directml.txt
```

Later, on the actual Snapdragon device (native ARM64 Python 3.11.x only):

```bash
pip uninstall onnxruntime onnxruntime-directml -y
pip install -r requirements-arm64-npu.txt
```

Run the backend:

```bash
uvicorn app.main:app --reload --port 8000
```

Test it:

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/hardware/info
curl http://127.0.0.1:8000/api/hardware/verify
```

**Expected output** for `/api/hardware/info` on a plain x64 dev machine with
only `onnxruntime` installed:
```json
{"available_providers": ["CPUExecutionProvider"], "npu_provider_present": false, "gpu_dml_provider_present": false, "cpu_provider_present": true, ...}
```
After installing `onnxruntime-directml`, `gpu_dml_provider_present` should
flip to `true` (assuming a DirectX 12 capable GPU is present) — verify with
`/api/hardware/verify`, which actually runs an op and reports whether it
was really executed on that provider.

**Common errors:**
- `ModuleNotFoundError: No module named 'onnx'` on `/api/hardware/verify` →
  `pip install onnx` (needed only for the in-memory micro-benchmark graph).
- Both `onnxruntime` and `onnxruntime-directml` installed at once → they
  conflict (same import name); uninstall one before installing the other.
- `onnxruntime-qnn` import errors on anything other than native ARM64
  Windows Python 3.11.x → expected; QNN EP simply won't appear in
  `get_available_providers()` elsewhere, and the engine will fall back
  automatically.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open the printed local URL (usually `http://localhost:5173`). The Dashboard,
Hardware Info, and Performance pages will work immediately as long as the
backend is running on port 8000. Other nav items show a "coming in Phase N"
placeholder — that's intentional, not a bug.

**Common errors:**
- Hardware/Performance pages show a red "could not reach backend" message →
  start the backend first, and confirm no firewall is blocking
  `localhost:8000`.
- CORS errors in the browser console → confirm you're hitting the frontend
  at `http://localhost:5173` (already whitelisted in
  `backend/app/core/config.py`); add your actual dev URL there if different.

## Common errors (Phase 1 additions)

- `RuntimeError: Tesseract binary not found` → install the system package
  above, not just `pytesseract` (which is only a Python wrapper). As of
  Phase 2 this no longer crashes the whole backend — see the robustness
  fix above — but OCR-dependent endpoints will 503 until it's installed.
- `pdf2image.exceptions.PDFInfoNotInstalledError` → `poppler-utils` (Linux)
  or the poppler binaries (Windows/macOS) aren't on PATH.
- OCR text looks garbled on messy backgrounds/diagrams → expected with the
  current simplified region-proposal step in `paddle_ocr_onnx.py` (and even
  Tesseract struggles on non-text-heavy images); a real DB detector export
  is the documented follow-up in `scripts/export_ocr_model.py`.
- Camera page shows a permissions error → `getUserMedia` requires HTTPS or
  `localhost` exactly (not a LAN IP) in most browsers.

## Common errors (Phase 2 additions)

- Study Vision page shows "LLM unavailable" under the AI answer → expected
  until you run `scripts/export_llm_model.py` and restart the backend; OCR
  results above it are still real and unaffected.
- `ModuleNotFoundError: No module named 'onnxruntime_genai'` → you're
  hitting a code path that needs it before installing it; `pip install
  onnxruntime-genai` (or the `-directml` variant) per the export script.
- Both `onnxruntime-genai` and `onnxruntime-genai-directml` installed at
  once → same import-name conflict as the raw `onnxruntime` pair;
  uninstall one first.
- LLM answers look cut off → raise `max_new_tokens` (sent from the
  frontend, capped server-side at 1024 in `app/api/vision.py`).

## Common errors (Phase 3 additions)

- Voice Assistant page shows "STT: unavailable" → install the always-on
  fallback (`pip install SpeechRecognition pocketsphinx`, plus `swig` as
  a system package if the wheel needs to build) or export a Whisper
  model (`scripts/export_stt_model.py`).
- Voice Assistant page shows "TTS: unavailable" → on Linux, install
  `espeak-ng` (`pyttsx3` needs a native engine to wrap); on Windows/macOS
  this should work out of the box via SAPI5/NSSpeechSynthesizer.
- `/api/voice/transcribe` or `/api/voice/ask` return 400 "Could not
  decode audio" → install `ffmpeg` and make sure it's on PATH; this is
  needed to decode whatever format the browser's `MediaRecorder`
  produced (typically webm/opus).
- `ModuleNotFoundError: No module named 'librosa'` / `'transformers'` →
  you're hitting the ONNX Whisper path before installing its extra
  deps -- `pip install transformers librosa soundfile` (see
  `scripts/export_stt_model.py`).
- `ModuleNotFoundError: No module named 'piper_phonemize'` → same idea
  for the ONNX Piper TTS path -- `pip install piper-phonemize` (see
  `scripts/export_tts_model.py`).
- Recorded audio produces empty/garbled transcripts with the PocketSphinx
  fallback → expected; it's a classical, lower-accuracy CPU engine by
  design (see `pocketsphinx_stt.py`'s docstring). Export the Whisper ONNX
  model for meaningfully better accuracy.
- Voice Assistant's mic button does nothing / permissions error →
  `getUserMedia` requires HTTPS or `localhost` exactly, same rule as the
  Camera Mode page.

## Common errors (Phase 4 additions)

- Settings page shows a provider as "not installed" and won't let you
  select it → that's intentional (see the honesty note in "Phase 4:
  what's new" above) — export the corresponding model
  (`export_llm_model.py` / `export_stt_model.py` / `export_tts_model.py`)
  or install the CPU fallback deps, then refresh Settings.
- After changing a provider preference, `live_status` in the Settings
  response still shows the old engine → the reload only re-runs that one
  service's backend search; if you forced a provider that isn't
  installed, `available` will be `false` with a reason naming the forced
  choice — that's the override working as intended, not a bug.
- History's "Replay" button sounds different from what you remember
  hearing live → expected. No audio is stored (see Phase 4 notes above);
  replay re-synthesizes the saved text through whichever TTS backend/voice
  is active *now*, which may differ from what was active when the turn
  happened.
- `backend/model_weights/app_state.db` doesn't exist yet on first run →
  it's created automatically on backend startup (see `app/storage/db.py`);
  no manual setup needed.

## Common errors (Phase 5 additions)

- Opening `http://localhost:8000/` shows the JSON `{"status":
  "backend-only", ...}` message instead of the UI → no frontend
  production build was found. Run `npm run build` in `frontend/` (or
  `deploy\setup.ps1` on Windows), then restart the backend.
- Refreshing on `/history` or `/settings` in production mode 404s →
  shouldn't happen; that's exactly what the catch-all route in
  `app/main.py` is for. If it does, confirm `frontend/dist/index.html`
  actually exists and you restarted the backend after the last build.
- `check_deployment.py` reports the FAIL "x64 Python running under
  Windows-on-ARM emulation" → install a native ARM64 Python 3.11.x build
  (not the regular x64 installer) and recreate the venv with it; this
  is the #1 way NPU support silently never works on the Snapdragon PC.
- `deploy\setup.ps1` / `deploy\run.ps1` refuse to run → PowerShell's
  default execution policy blocks unsigned local scripts; run them as
  `powershell -ExecutionPolicy Bypass -File deploy\setup.ps1` (as shown
  in each script's own header), rather than double-clicking them.
- After running `deploy\setup.ps1`, `/api/hardware/verify` still shows
  CPU only → the base `requirements.txt` is deliberately CPU-safe; you
  still need the manual DirectML or QNN install step the script prints
  at the end, then restart the server.

## Next phase (Phase 4, once you confirm this works for you)

Conversation History (persisted across sessions, browsable, replayable)
and Settings (model selection, provider preference overrides, voice
selection) — the last two placeholder pages in the sidebar.

## Phase 4: what's new (Conversation History + Settings)

- `app/storage/` — a new, separate SQLite file (`backend/model_weights/app_state.db`,
  distinct from the benchmarking DB) backing two features:
  - **Conversation History** (`conversation_store.py`) — every completed
    Study Vision and Voice Assistant turn is saved automatically: OCR
    text/transcript, question, AI answer (if the LLM was available), and
    a JSON blob of every stage's actual engine/provider/latency. Logging
    is best-effort — a storage hiccup is caught and logged, never lets a
    real request fail.
  - **Settings** (`settings_store.py`) — a single persisted JSON document
    (provider preferences, default mode, default token budget, TTS
    voice, auto-speak toggle).
- `/api/history/*` (`app/api/history.py`) — list (with kind filter),
  get-one, delete-one, delete-all, plus `POST /{id}/replay-audio`.
  **Important honesty note:** raw audio is *not* stored (keeps this a
  lightweight text log, not a growing media archive), so "replay" means
  re-synthesizing the *saved answer text* through whichever TTS backend
  is active right now — not playing back the original clip. The History
  page says this explicitly next to the replay button.
- `/api/settings` (`app/api/settings.py`) — `GET` returns current
  settings plus what's actually available right now, live: `available.*`
  is built by calling each service's new `available_backends()`, so the
  Settings page can never offer a provider that isn't really installed.
  `PUT` validates and persists a partial patch, then reloads only the
  services whose setting changed. `POST /reset` restores defaults.
- **Provider preference overrides, done honestly**: `LLMService`,
  `STTService`, and `TTSService` all gained a `reload(preference=...)`
  method. Forcing a specific provider (e.g. `tts_backend_preference:
  "pyttsx3"`) makes that service skip the others entirely — if the
  forced one isn't installed, it reports unavailable *naming the forced
  choice*, instead of silently falling back to whatever else works. An
  override that quietly no-ops isn't an override.
- `pyttsx3_tts.py` gained voice selection: `Pyttsx3TTS.list_voices()`
  reports every voice the native OS engine actually has installed (not a
  hardcoded list), and `voice_id` can be forced through Settings.
- Two new React pages, `History.jsx` and `Settings.jsx`, replacing their
  Phase 6 placeholders — both wired to the endpoints above, no other
  pages changed.

Tested end-to-end on this dev machine (no NPU, no exported LLM/STT/TTS
models — CPU/pyttsx3/pocketsphinx fallbacks only): settings PUT/reset,
forced-provider unavailability messaging, Study Vision and Voice turns
logging correctly, history list/detail/delete/clear, and replay-audio
producing a playable WAV. The LLM answer fields will be empty until you
export a model (`scripts/export_llm_model.py`), same as Phase 3 — History
still logs the OCR/transcript half of the turn either way.

## Next phase (Phase 5, once you confirm this works for you)

Not yet scoped — nothing past Settings was in the original roadmap.
Natural candidates: packaging/deployment for the actual Snapdragon PC,
or a polish pass once real NPU numbers are available.

## Phase 5: what's new (packaging & deployment)

Not a new feature page — this phase is about turning the four pieces
above into something you can actually hand to a judge or run on the
Snapdragon PC without two terminals and a mental checklist.

- **Single-process production serving** (`app/main.py`) — when
  `frontend/dist/` exists (`npm run build`), the FastAPI backend now
  mounts and serves it directly: one process, one port, no separate
  `npm run dev` needed for a demo. Client-side routes (`/history`,
  `/settings`, etc.) correctly resolve on a hard refresh, real static
  assets are served with the right content type, and — important — an
  unknown `/api/...` path still returns a genuine 404 instead of being
  silently swallowed by the SPA catch-all. Dev mode (`npm run dev` +
  `uvicorn --reload`, two ports, CORS) still works exactly as before;
  this is additive, not a replacement.
- **`scripts/check_deployment.py`** — a standalone preflight CLI, no
  server needed: `python scripts/check_deployment.py`. Reports platform/
  architecture (specifically catching x64 Python running under Windows-
  on-ARM emulation — it imports and runs fine, it just can never load
  QNNExecutionProvider, and that's otherwise invisible), required system
  binaries on PATH, which Python packages are importable, which ONNX
  Runtime execution providers are actually registered, which models are
  actually exported on disk, whether the frontend's production build
  exists, and free disk space. Exit codes: `0` fully ready, `1` ready
  with fallbacks (nothing faked, just CPU-only), `2` a real blocker.
- **`deploy/setup.ps1`** — one-shot setup for the target Windows ARM64
  machine: venv, base (CPU-safe) requirements, frontend production
  build, then runs the preflight check above. Deliberately does NOT
  auto-install the DirectML or QNN requirement variants — each one
  *replaces* a conflicting base package (see `requirements-directml.txt`
  / `requirements-arm64-npu.txt`), so that stays a manual, deliberate
  step; the script prints the exact commands at the end.
- **`deploy/run.ps1`** — starts the packaged app on
  `http://localhost:8000` (configurable via `-Port`) and opens a browser
  tab automatically.

Tested end-to-end on this dev machine (Linux x64, no NPU/DirectML):
`check_deployment.py` correctly reports CPU-only fallbacks with exit
code 1 before a frontend build exists, and exit code 1 (all WARN, no
FAIL) after building it; the production single-process server was
started for real and verified to serve `index.html` at `/`, a client
route (`/history`) on direct hit, a real built JS asset with the
correct content type, `/api/health` and `/api/settings` through the
same mount, and a genuine `404 {"detail": "Unknown API route: ..."}`
for an unknown `/api/...` path rather than silently serving the SPA
shell. The `.ps1` scripts themselves could only be reviewed, not
executed — no Windows/PowerShell in this sandbox — so give them a real
run on the target machine and report back anything that doesn't match.

## Next phase (Phase 6, once you confirm this works for you)

Not yet scoped — everything in the original roadmap plus deployment
packaging is now done. The one thing that still can't be verified from
here: an actual run on the Snapdragon PC itself, confirming
`/api/hardware/verify` really reports `QNNExecutionProvider` once
`requirements-arm64-npu.txt` is installed there.
