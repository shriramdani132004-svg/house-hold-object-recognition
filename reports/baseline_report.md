# Phase 5 — Baseline Model Report

Zero-shot evaluation of the COCO-pretrained lightweight YOLO model
on the project test split, before any custom training (Phase 6).
Generated 2026-10-01T19:41:23+00:00 by `scripts/evaluate_baseline.py`.

## 1. Overview

| Item | Value |
|------|-------|
| Model | yolo26n (COCO-pretrained, no fine-tuning) |
| Test images | 1965 |
| Ground-truth instances | 9174 |
| Classes | 19 household objects |
| Confidence threshold | 0.25 |
| Precision | 0.6924 |
| Recall | 0.3914 |
| F1 | 0.5001 |
| mAP@0.5 | 0.5485 |
| mAP@0.5:0.95 | 0.3961 |
| Speed (CPU, warm average) | 418.0 ms/image |

## 2. Model

- Weights: `models/baseline/yolo26n.pt` — yolo26n checkpoint,
  checkpoint saved 2025-12-15 (ultralytics 8.3.222).
- Architecture: YOLO26-nano detection model from Ultralytics with an
  end-to-end (NMS-free) head; 80 COCO output classes.
- Runtime: ultralytics 8.4.171,
  torch 2.14.1+cpu, device `cpu` (CPU only —
  no CUDA GPU on this machine).
- Pretrained on the COCO detection dataset; **not** trained or
  fine-tuned on project data. Results are a true zero-shot baseline.
- Checkpoint license: AGPL-3.0 (Ultralytics).

## 3. Dataset and Class Mapping

- Data: `data/processed/household_objects` test split —
  1965 images / 9174 instances
  across 19 classes (train/val were **not** touched).
- Mapping: name-based — each model COCO class name that equals a
  project class name in `configs/classes.yaml` is mapped to that
  project class id.
- Classes mapped: 19 / 19 project classes present in the model's output space.
- Detections at conf >= 0.01: 45926 of 87186 raw predictions belong to the 19 project
  classes; 41260 (47.3%) of other COCO classes were
  ignored (they are not counted as false positives).

## 4. Evaluation Protocol

- Single streamed inference pass over the test split,
  `imgsz=640`, `conf>=0.01`,
  `max_det=100`, CPU; all thresholds below are
  computed offline from that one pass.
- Detection matching: greedy one-to-one per image and class in
  descending confidence order; a true positive needs same class and
  IoU >= 0.50.
- Precision / recall / F1 are micro-averaged over the whole split at
  the primary threshold (0.25).
- mAP is implemented in-repo (`src/evaluation/baseline.py`):
  score-ordered greedy matching, COCO-style 101-point interpolated
  precision envelope; mAP@0.5 = AP at IoU 0.50, mAP@0.5:0.95 = mean
  AP over IoU 0.50:0.05:0.95; per-class AP is `None` when the class
  has no ground truth and is excluded from the mAP mean. Classes
  without ground truth therefore do not lower the reported mAP.
- Evaluation is class-restricted by design: only the 19 project
  classes participate, so COCO-only objects (e.g. person) neither
  help nor hurt the metrics.

## 5. Overall Metrics

| Metric | Value |
|--------|-------|
| TP / FP / FN | 3591 / 1595 / 5583 |
| Precision | 0.6924 |
| Recall | 0.3914 |
| F1 | 0.5001 |
| mAP@0.5 | 0.5485 |
| mAP@0.5:0.95 | 0.3961 |

## 6. Per-Class Metrics

| Class | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50-95 |
|-------|----|----|----|----|-----------|--------|----|------|---------|
| chair | 1771 | 590 | 277 | 1181 | 0.6805 | 0.3331 | 0.4473 | 0.4345 | 0.2786 |
| book | 1129 | 93 | 77 | 1036 | 0.5471 | 0.0824 | 0.1432 | 0.2155 | 0.1080 |
| bottle | 1013 | 371 | 164 | 642 | 0.6935 | 0.3662 | 0.4793 | 0.4572 | 0.3064 |
| cup | 895 | 385 | 212 | 510 | 0.6449 | 0.4302 | 0.5161 | 0.4914 | 0.3542 |
| dining table | 695 | 316 | 185 | 379 | 0.6307 | 0.4547 | 0.5284 | 0.4886 | 0.3535 |
| bowl | 623 | 298 | 141 | 325 | 0.6788 | 0.4783 | 0.5612 | 0.5467 | 0.4057 |
| potted plant | 342 | 122 | 80 | 220 | 0.6040 | 0.3567 | 0.4485 | 0.4357 | 0.2671 |
| wine glass | 341 | 124 | 43 | 217 | 0.7425 | 0.3636 | 0.4882 | 0.4373 | 0.2856 |
| cell phone | 262 | 89 | 37 | 173 | 0.7063 | 0.3397 | 0.4588 | 0.4433 | 0.3041 |
| clock | 267 | 170 | 25 | 97 | 0.8718 | 0.6367 | 0.7359 | 0.6944 | 0.4912 |
| tv | 288 | 188 | 45 | 100 | 0.8069 | 0.6528 | 0.7217 | 0.7238 | 0.5567 |
| couch | 261 | 154 | 65 | 107 | 0.7032 | 0.5900 | 0.6417 | 0.6166 | 0.4502 |
| remote | 283 | 73 | 53 | 210 | 0.5794 | 0.2580 | 0.3570 | 0.3445 | 0.2110 |
| sink | 225 | 112 | 42 | 113 | 0.7273 | 0.4978 | 0.5910 | 0.5644 | 0.3746 |
| laptop | 231 | 158 | 37 | 73 | 0.8103 | 0.6840 | 0.7418 | 0.7218 | 0.6033 |
| bed | 163 | 100 | 38 | 63 | 0.7246 | 0.6135 | 0.6645 | 0.6663 | 0.4987 |
| keyboard | 153 | 91 | 38 | 62 | 0.7054 | 0.5948 | 0.6454 | 0.6451 | 0.5074 |
| refrigerator | 126 | 83 | 18 | 43 | 0.8218 | 0.6587 | 0.7313 | 0.7556 | 0.6184 |
| mouse | 106 | 74 | 18 | 32 | 0.8043 | 0.6981 | 0.7475 | 0.7381 | 0.5518 |

## 7. Confidence Threshold Analysis

| Confidence | Precision | Recall | F1 | TP | FP | FN |
|------------|-----------|--------|----|----|----|----|
| 0.25 | 0.6924 | 0.3914 | 0.5001 | 3591 | 1595 | 5583 |
| 0.50 | 0.8686 | 0.2644 | 0.4054 | 2426 | 367 | 6748 |
| 0.75 | 0.9575 | 0.1349 | 0.2366 | 1238 | 55 | 7936 |

Full write-up: `reports/baseline_confidence_analysis.md`.

## 8. Inference Speed

| Item | Value |
|------|-------|
| Device | cpu (no CUDA GPU available) |
| Images processed | 1965 |
| Total inference time | 816.63 s |
| Average wall time | 415.6 ms/image |
| Warm average (first 20 excluded) | 418.0 ms/image |
| Model-reported inference | 356.6 ms/image |
| Throughput | 2.41 images/s |
| Model load (one time) | 41.3 ms |
| First image (warmup) | 1640.1 ms |

Wall time includes decode, pre- and post-processing; model-reported
time is the kernel-only figure from Ultralytics. Figures are from
this machine and will differ elsewhere.

## 9. Error Analysis Summary

| Category | Count |
|----------|-------|
| low_confidence | 2856 |
| poor_localization | 2183 |
| missed_detection | 1001 |
| false_positive | 901 |
| cluttered_scene | 634 |
| wrong_class | 119 |

- Missed ground-truth objects: 5583 (small objects: 3674)
- Images with at least one failure: 1587 / 1965
- Top class confusion: bottle -> cup (18x)

Details and example IDs: `reports/baseline_error_examples.md`.

## 10. Sample Review

20 annotated test images (ground truth + predictions, deterministic selection, seed 42) in `reports/baseline_samples/` with `index.md`.

1. `01_crowded_000000031296.jpg` — crowded: most ground-truth objects (38) (GT 38, predicted 12)
2. `02_single_object_000000000776.jpg` — single_object: exactly one ground-truth object (GT 1, predicted 0)
3. `03_small_object_000000522393.jpg` — small_object: smallest ground-truth box (area 0.000023) (GT 1, predicted 0)
4. `04_large_object_000000010583.jpg` — large_object: largest ground-truth box (area 1.000) (GT 1, predicted 2)
5. `05_class_chair_000000000139.jpg` — class_chair: contains 'chair' (GT 13, predicted 11)
6. `06_class_book_000000000632.jpg` — class_book: contains 'book' (GT 17, predicted 4)
7. `07_class_bottle_000000002685.jpg` — class_bottle: contains 'bottle' (GT 12, predicted 0)
8. `08_class_cup_000000002157.jpg` — class_cup: contains 'cup' (GT 12, predicted 10)
9. `09_class_dining_table_000000001993.jpg` — class_dining_table: contains 'dining table' (GT 4, predicted 3)
10. `10_class_bowl_000000001425.jpg` — class_bowl: contains 'bowl' (GT 1, predicted 2)
11. `11_class_potted_plant_000000013923.jpg` — class_potted_plant: contains 'potted plant' (GT 17, predicted 10)
12. `12_class_wine_glass_000000002431.jpg` — class_wine_glass: contains 'wine glass' (GT 4, predicted 5)
13. `13_class_cell_phone_000000001268.jpg` — class_cell_phone: contains 'cell phone' (GT 1, predicted 0)
14. `14_class_clock_000000001296.jpg` — class_clock: contains 'clock' (GT 2, predicted 1)
15. `15_random_000000021503.jpg` — random: seeded random pick (seed 42) (GT 3, predicted 2)
16. `16_random_000000355325.jpg` — random: seeded random pick (seed 42) (GT 6, predicted 6)
17. `17_random_000000507893.jpg` — random: seeded random pick (seed 42) (GT 1, predicted 1)
18. `18_random_000000513041.jpg` — random: seeded random pick (seed 42) (GT 6, predicted 5)
19. `19_random_000000507235.jpg` — random: seeded random pick (seed 42) (GT 3, predicted 3)
20. `20_random_000000323151.jpg` — random: seeded random pick (seed 42) (GT 4, predicted 4)

## 11. Limitations

- The baseline has never seen the project's curated 19-class data;
  per-class AP mostly reflects COCO coverage of that class name.
- mAP is computed by the in-repo implementation (pycocotools is
  not installed); conventions are documented in section 4.
- Evaluation is class-restricted: detections of non-project COCO
  classes are ignored rather than penalized.
- Timings are CPU-only on one machine; `torch.set_num_threads` was
  left at its default.
- The confidence table reuses one pass, so thresholds below the
  capture floor (0.01) are not observable.

## 12. Reproduction and Next Steps

Commands (from the project root):

```powershell
python scripts\evaluate_baseline.py
python scripts\baseline_inference.py --source <image> --save
```

Artifacts: `reports/baseline_results.json` (machine-readable),
`reports/baseline_confidence_analysis.md`,
`reports/baseline_error_examples.md`,
`reports/baseline_samples/`,
`reports/baseline_commands.md`.

Next: Phase 6 trains a custom model on the train split and
compares against these numbers on the same test split.
