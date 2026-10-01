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

## Project Status

**In progress — Phase 1 (Project Setup) and Phase 2 (Dataset) complete.**

- [x] Project structure, virtual environment, Git, documentation
- [x] Dataset selection, acquisition & documentation
- [ ] Data preparation, class selection & splits
- [ ] Baseline & model training
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
| Deep learning | PyTorch + Ultralytics YOLO (added in training phase) |
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
