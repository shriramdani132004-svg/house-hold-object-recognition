# Household Object Recognition

An object-recognition project that learns common household objects — from
dataset preparation and continual-learning training to a phone-first Gradio
web app where you recognize objects by live camera or photo upload and get
the object name with a confidence score.

## Dataset

- **Selected dataset:** CORe50 — Continual Recognition of 50 Objects
  (128x128, sessions s1–s11, 50 objects / 10 categories, 164,866 PNGs)
- **Official source:** <https://vlomonaco.github.io/core50/> ·
  <https://github.com/vlomonaco/core50>
- **Purpose:** primary assignment dataset for continual recognition —
  sequential sessions drive the NI/NC/NIC experience splits with
  test sessions s3/s7/s10.
- **Full details** (license, size, classes, annotation format, acquisition
  method, citation): see `docs/dataset/core50/DATASET_INFO.md` and
  [DATASET.md](DATASET.md).

Acquired 2026-10-02 under `data/raw/core50/` (git-ignored) with SHA-256
manifest in `data/raw/core50/downloads/MANIFEST.json`; validated by
`scripts/validate_core50.py` (summary:
`reports/phase2_core50_summary.json`, status PASS).

> **Prototype note:** the earlier COCO 2017 phases (1–6 of the original
> roadmap) remain intact under `data/raw/coco/`,
> `data/processed/household_objects/` and `models/` as the static-detection
> prototype — see [DATASET.md](DATASET.md).

## Data Preparation (Phase 3)

The prepared YOLO-format dataset lives in `data/processed/household_objects/`
(git-ignored, regenerable):

- **19 household classes** (IDs 0–18) defined in `configs/classes.yaml`;
  selection rationale in `reports/phase3_class_selection.md`.
- **47,399 images / 214,983 instances** — train 40,890 · val 4,544 ·
  test 1,965 (COCO train2017 split 90/10 with seed 42; COCO val2017 → test).
- Validation report: `reports/phase3_dataset_validation.md` (PASS).
- Statistics and figures: `reports/phase3_dataset_statistics.md`,
  `reports/figures/phase3_*.png`.

Rebuild everything from the project root (project venv on Windows):

```bash
.\.venv\Scripts\python.exe scripts/analyze_categories.py
.\.venv\Scripts\python.exe scripts/prepare_dataset.py
.\.venv\Scripts\python.exe scripts/validate_prepared_dataset.py
.\.venv\Scripts\python.exe scripts/visualize_prepared_dataset.py
.\.venv\Scripts\python.exe scripts/dataset_statistics.py
```

## Dataset Analysis (Phase 4)

Read-only analysis of the prepared dataset produced by
`scripts/analyze_dataset.py` (a few minutes: label scan, image-header scan,
SHA-1 duplicate check):

- **Full report:** `reports/phase4_dataset_analysis.md` — 17 sections covering
  class distribution, imbalance, objects per image, box sizes, co-occurrence,
  split proportions, dimensions, annotation quality, and concerns;
  machine-readable summary in `reports/phase4_summary.json`.
- **Key findings:** class imbalance 16.8x (chair 39,842 vs mouse 2,367
  instances); 49.5% of boxes are small (< 1% of image area); 67.8% of images
  contain 2+ target objects; zero malformed or missing annotations; no
  cross-split duplicates (6 within-train duplicate pairs).
- **Figures:** `reports/figures/phase4/` (8 charts).
- **Samples:** `reports/phase4_samples/` — 27 annotated examples (one per
  class plus crowded, small/large box, extreme-aspect, and split cases).

```bash
.\.venv\Scripts\python.exe scripts/analyze_dataset.py
```

## Baseline Model (Phase 5)

Zero-shot baseline: the official COCO-pretrained **YOLO26n** checkpoint run
on the 1,965-image test split (no training, no fine-tuning). Full report:
`reports/baseline_report.md`.

- **Metrics** (conf >= 0.25, IoU >= 0.5): precision **0.6924**, recall
  **0.3914**, F1 **0.5001**, mAP@0.5 **0.5485**, mAP@0.5:0.95 **0.3961**
  (TP 3,591 / FP 1,595 / FN 5,583).
- **Speed** (CPU, no CUDA GPU): 416 ms/image average (418 ms warm),
  816.63 s total for 1,965 images, 2.41 images/s.
- Class-restricted evaluation with an explicit name-based COCO -> 19-class
  mapping; metric conventions are documented in report section 4.
- Companion documents: `reports/baseline_confidence_analysis.md`,
  `reports/baseline_error_examples.md`, `reports/baseline_commands.md`,
  machine-readable `reports/baseline_results.json`, and 20 annotated
  samples in `reports/baseline_samples/`.
- Weights `models/baseline/yolo26n.pt` are git-ignored and auto-downloaded
  on first run; model card: `models/baseline/README.md`.

```bash
.\.venv\Scripts\python.exe scripts/evaluate_baseline.py
.\.venv\Scripts\python.exe scripts/baseline_inference.py --source <image> --save
```

## Custom Model Training (Phase 6)

One training experiment: **YOLO26n**, fine-tuned on the project's 19-class
dataset (`train` split only, validation split for model selection, test
split untouched). Full report: `reports/phase6_training_report.md`.

- **Schedule**: 2 epochs, batch 16, imgsz 640, seed 42, deterministic,
  warmup 0.5 epochs, patience 15 — chosen up front for the CPU-only
  machine (~6.9 h/epoch measured; 46,086.5 s total wall time), not tuned
  against results. Data config `configs/train_data.yaml` contains no
  `test` key.
- **Result** (validation split, 4,544 images / 21,100 instances, best.pt):
  precision **0.621**, recall **0.501**, mAP@0.5 **0.541**,
  mAP@0.5:0.95 **0.378** — a validation-split result only, not final
  test performance (Phase 7 does the held-out comparison).
- Dataset integrity verified pre/post: train 40,890 / val 4,544 /
  test 1,965 images, 214,983 instances — unchanged.
- Artifacts: weights `models/training/household_yolo26n/weights/
  {best,last}.pt` (git-ignored), learning curves and confusion matrix in
  `reports/figures/phase6/`, machine-readable
  `reports/phase6_training_results.json`, summary
  `reports/phase6_training_summary.md`.
- CPU benchmarks of threads/workers/channels_last/compile were neutral
  (±5%); Ultralytics CPU defaults kept. Speed levers and probe commands:
  `reports/phase6_training_commands.md`.

```bash
.\.venv\Scripts\python.exe scripts/train_model.py
.\.venv\Scripts\python.exe scripts/validate_training.py
.\.venv\Scripts\python.exe scripts/benchmark_training_speed.py --tag c
```

## Inference System + Phone Web App (Phase 7)

The frozen Phase-6 selection — `models/continual/final_model.pt`
(Experience Replay, SmallConvNet, 50 CORe50 classes, 64x64 input,
SHA-256 `b2f0606c…e351`) — is wrapped in a reusable inference layer and
a simple phone-first Gradio application. The checkpoint is loaded **once
per process** and reused for every frame (`eval()` + `torch.no_grad()`,
CPU only); no retraining, no checkpoint changes.

- **Reusable API** (`src/inference/`): `get_engine()` returns the cached
  `InferenceEngine`; `predict(...)` / `predict_pil(...)` /
  `predict_numpy(...)` / `predict_frame(...)` accept PIL images, NumPy
  RGB/RGBA/grayscale arrays, and camera frames — all through one
  preprocessing path that mirrors Phase-5 training exactly (RGB,
  bilinear resize to 64x64, `/255`, mean/std 0.5). A prediction carries
  the object name, confidence, and class index
  (`Prediction.format_block()` → `OBJECT` / `CONFIDENCE` / `CLASS ID`).
- **Live camera + image upload:** one Gradio page in `app/` with a single
  shared callback for both inputs; invalid input returns a readable error
  instead of crashing.
- **Local launch** (from the project root):

  ```bash
  .\.venv\Scripts\python.exe -m app
  ```

  Optional public sharing (the public URL is **not** verified here —
  actual deployment belongs to Phase 8): set `GRADIO_SHARE = "true"`
  before launching.
- **Current model limitation:** the selected model is a *classifier* —
  it predicts object name + confidence only. **Bounding boxes are not
  available** for it and are never drawn. Phone-camera validation and
  public deployment are Phase 8 steps.

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_phase7_inference.py -q
```

## Project Status

**Phase 7 (reusable inference system + phone-first web app) complete — next: Phase 8 practical testing + Hugging Face deployment.**

- [x] Project structure, virtual environment, Git, documentation
- [x] CORe50 dataset acquisition, integrity validation & documentation
- [x] COCO prototype: data preparation, analysis, baseline, training
- [x] Continual data pipeline (NI/NC/NIC experiences)
- [x] Naive baseline + experience replay
- [x] Main NIC experiment + evaluation (per-experience accuracy/forgetting, comparison, plots, report)
- [x] Error/forgetting analysis & final model selection
- [x] Inference system & web app
- [ ] Testing, deployment, final documentation

## Planned Features

- Curated public household-object detection dataset with clean train/val/test splits
- Custom-trained lightweight YOLO detection model
- Evaluation suite: precision, recall, mAP50, mAP50-95, confusion matrix, speed
- Error analysis and iterative improvement loop
- Reusable Python inference API (image → boxes, classes, confidence scores, object summary)
- Gradio web app for interactive image upload and detection
- Automated test suite and CI via GitHub Actions
- Deployment to Hugging Face Spaces

## Planned Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.14 |
| Deep learning | PyTorch + Ultralytics YOLO (CPU, added in Phase 5) |
| Data handling | NumPy, Pillow, pandas |
| Visualization | Matplotlib, seaborn |
| Web app | Gradio |
| Testing | pytest, GitHub Actions CI |
| Deployment | Hugging Face Spaces |

> Dependencies are installed per phase; heavy ML libraries are not part of
> the setup phase.

## Project Structure

```
house_hold_obj/
├── app/                  # Gradio web application
├── configs/              # YAML configuration files
├── data/
│   ├── raw/              # Original downloaded dataset
│   ├── processed/        # Cleaned / converted data
│   └── splits/           # Train/val/test splits
├── examples/
│   ├── input/            # Sample input images
│   └── output/           # Sample detection outputs
├── models/               # Trained weights & checkpoints
├── notebooks/            # Exploration & analysis notebooks
├── reports/
│   ├── figures/          # Plots and figures
│   └── ...               # Evaluation reports
├── scripts/              # Runnable pipeline scripts
├── src/
│   ├── data/             # Data loading & preparation
│   ├── training/         # Training loop & experiments
│   ├── evaluation/       # Metrics & evaluation
│   ├── inference/        # Prediction API
│   └── utils/            # Shared helpers
├── tests/                # pytest test suite
├── .github/workflows/    # CI configuration
├── AGENTS.md             # AI-agent project instructions
├── PROJECT_PLAN.md       # Full phase-by-phase roadmap
├── requirements.txt      # Python dependencies
├── .gitignore
└── LICENSE
```

## Getting Started (once dependencies grow)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## License

See [LICENSE](LICENSE).
