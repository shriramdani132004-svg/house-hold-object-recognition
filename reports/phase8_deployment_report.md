# Phase 8 Deployment Report

## Environment

- Local Python: 3.14.2 (Windows, CPU-only) — torch 2.14.1+cpu, gradio 6.29.1
- Hugging Face account namespace: `shriram132004` (verified with `hf auth whoami`)
- Deployment date: 2026-10-03

## Frozen Model

- Path: `models/continual/final_model.pt`
- SHA-256: `b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351`
- Status: **PASS** — hash verified before and after Phase 8 work; never retrained,
  replaced, or modified. `models/continual/final_model.json` present and its
  recorded hash matches.

## Local Verification

- Phase 7 focused tests (`tests/test_phase7_inference.py`): **32 passed**
  (31 original + 1 new `set_engine` deployment-hook test)
- Full pytest (`tests -q`): **272 passed**
- compileall (`src scripts tests app deployment`): **exit 0**
- Local inference smoke (web-app callback path, one real CORe50 image
  `data/raw/core50/dataset/core50_128x128/s1/o1/C_01_01_000.png`):
  - OBJECT `plug_adapter1`, CONFIDENCE `99.99%`, CLASS ID `0`
  - invalid input (None) → controlled user-facing error, no crash
  - 50-name class mapping resolved — **PASS**
  - (plumbing smoke test only; not an accuracy measurement)
- Deployment package local run: `app.py` entry loaded, engine wired
  (1 checkpoint load, 50 classes), callback predicted `plug_adapter1`
  (`99.99%`), invalid input handled, and a local launch on
  `127.0.0.1:7899` served the Gradio page (23,695 bytes HTML) — **PASS**

## Hugging Face

- Intended Space name: `continual-household-object-recognition`
- Intended repository: `shriram132004/continual-household-object-recognition`
- Space URL: **NOT CREATED**
- SDK: gradio (intended; README YAML prepared with `sdk: gradio`,
  `sdk_version: 6.29.1`, `app_file: app.py`, `python_version: "3.11"`)
- Hardware: `list_spaces_hardware()` listed `cpu-basic` ($0/min) and
  `zero-a10g` ZeroGPU ($0/min) among account options, but actual Gradio
  Space creation is entitlement-blocked (below)
- Build result: **NOT RUN** (repository could not be created)
- Runtime result: **NOT RUN**

Exact API errors captured (no retries after the entitlement failures; no
billing action taken; no secrets printed):

1. Gradio Space on `cpu-basic` →
   `HTTP 402 Payment Required` (POST https://huggingface.co/api/repos/create):
   "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
   on free cpu-basic requires a PRO subscription. Subscribe at
   https://huggingface.co/pro"
2. Gradio Space on `zero-a10g` (ZeroGPU) →
   `HTTP 402 Payment Required` (POST https://huggingface.co/api/repos/create):
   "You must be subscribed to PRO to host Spaces with ZeroGPU. If you recently
   created your account, please wait 30 days or request a community grant.
   Visit your billing settings to subscribe:
   https://huggingface.co/settings/billing"

## Remote Verification

- Remote page result: **NOT RUN** (no Space exists; confirmed `404` via
  `space_info` after the failed creation attempts)
- Remote inference result: **NOT RUN**
- Actual test input / predicted label / confidence: **NOT AVAILABLE**
- Invalid-input remote result: **NOT RUN**

## Camera

- Camera interface in the deployment package: present
  (`gr.Image(sources=["webcam", "upload"])`, one shared callback with upload)
  — verified during the local deployment-page check
- Camera interface deployed: **FAIL** (no Space could be created)
- Physical phone test: **NOT TESTED** — no physical device available to
  automation in this environment

## Deployment Package (prepared and verified locally)

Location: `deployment/huggingface_space/`

- `README.md` — Hugging Face YAML (`sdk: gradio`, `sdk_version: 6.29.1`,
  `app_file: app.py`, `python_version: "3.11"`)
- `app.py` — Space entrypoint: wires the frozen checkpoint + official class
  mapping into the shared Phase-7 `InferenceEngine` singleton via
  `set_engine()`, then serves the Phase-7 UI
- `requirements.txt` — `gradio==6.29.1`, `torch==2.14.1+cpu` (CPU wheel
  index), `pillow`, `numpy`, `pyyaml` (dependency closure of the runtime
  imports; no ultralytics/matplotlib/tqdm)
- `app/` + `src/` (25 files: inference, training, data, evaluation/continual;
  excludes `baseline.py`, `error_analysis.py`, `src/utils`)
- `models/continual/{final_model.pt, final_model.json, object_mapping.json}`
- Excludes: all datasets, `.venv`, caches, logs, reports, credentials —
  package precheck found no secrets and no dataset directories

## Limitations

- Public deployment is blocked by the Hugging Face account entitlement:
  hosting **Gradio Spaces** (cpu-basic) and **ZeroGPU Spaces** requires a
  **PRO subscription** on this account; free *Static* Spaces cannot execute
  a Gradio application, so they are not a valid alternative for remote
  inference.
- Required account action before Phase 8 can be re-attempted (either):
  1. Subscribe to Hugging Face PRO (https://huggingface.co/pro), **or**
  2. Follow HF's stated ZeroGPU alternative: wait 30 days (new account)
     or request a community grant, then re-run Phase 8.
- No public URL, remote prediction, phone-camera, or build/runtime
  measurement is claimed — none were obtained.
