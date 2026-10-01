# Household Object Recognition

An object-detection project that recognizes common household objects in
images — from dataset preparation and custom model training to a Gradio web
app where you upload a photo and get labeled bounding boxes with confidence
scores.

## Dataset

- **Selected dataset:** COCO — Common Objects in Context (COCO 2017 release)
- **Official source:** <https://cocodataset.org/> (download page & Terms of Use)
- **Purpose:** real-world images with professional bounding-box annotations
  for training and evaluating the household-object detector. The
  household-relevant class subset is finalized during data preparation.
- **Full details** (license, size, classes, annotation format, acquisition
  method, citation): see [DATASET.md](DATASET.md).

Raw data lives under `data/raw/coco/` (git-ignored); acquisition and
verification are handled by `scripts/download_dataset.py` and
`scripts/verify_dataset.py`. Verified metadata is recorded in
`data/raw/DATASET_INFO.txt`.

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
  817.71 s total for 1,965 images, 2.4 images/s.
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

## Project Status

**In progress — Phases 1–5 complete.**

- [x] Project structure, virtual environment, Git, documentation
- [x] Dataset selection, acquisition & documentation
- [x] Data preparation, class selection & splits
- [x] Dataset analysis & reporting
- [x] Baseline model (zero-shot evaluation)
- [ ] Custom model training
- [ ] Evaluation & improvement
- [ ] Inference system & web app
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
