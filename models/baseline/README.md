# Baseline model weights

This directory holds the pretrained checkpoint used for the **Phase 5
baseline** evaluation. The weights are **not committed** to git (`models/*`
and `*.pt` are git-ignored); this README is the tracked placeholder.

## What is stored here

| Item | Value |
|---|---|
| File | `yolo26n.pt` (about 5.3 MB) |
| Architecture | YOLO26-nano (YOLO26n), single-stage detector |
| Task | Object detection, 80 COCO classes |
| Checkpoint | Official Ultralytics COCO-pretrained release (end-to-end / NMS-free head) |
| Trained with | Ultralytics 8.3.222 on COCO (`coco.yaml`), 245 epochs, `imgsz=640` |
| Checkpoint date | 2025-12-15 |
| License | AGPL-3.0 — see <https://ultralytics.com/license> |
| Verified runtime | ultralytics 8.4.171 + torch 2.14.1+cpu (CPU only) |

The model outputs the standard 80 COCO class names. For this project the
detections are restricted to the 19 household classes by **name** (COCO
class name must equal the project class name in `configs/classes.yaml`);
see `src/evaluation/baseline.py` for the mapping logic.

## Role in the project

The baseline is evaluated **zero-shot** — no training or fine-tuning on
project data — against the 1,965-image test split, so later trained models
have a reference point. See `reports/baseline_report.md`.

## Getting the weight on a fresh clone

Because the `.pt` file is git-ignored, a fresh clone does not contain it.
Either script downloads it automatically on first run:

```powershell
.\.venv\Scripts\python.exe scripts\baseline_inference.py --source examples\input\your_image.jpg
.\.venv\Scripts\python.exe scripts\evaluate_baseline.py
```

Both resolve `model.path` from `configs/baseline.yaml` and, when the file
is missing, download the official `yolo26n.pt` via the `ultralytics`
package and place it at `models/baseline/yolo26n.pt`. Manual equivalent:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"
```

(requires network access; subsequent runs work fully offline).
