# Phase 5 — Baseline Confidence Threshold Analysis

All thresholds are computed offline from a single inference pass with
detections kept down to a confidence floor of 0.01, so no model
re-run is involved. Matching: greedy one-to-one, same class only,
IoU >= 0.5, over the 1965 evaluated test images.

| Confidence | Precision | Recall | F1 | TP | FP | FN | Predicted |
|------------|-----------|--------|----|----|----|----|-----------|
| 0.25 | 0.6924 | 0.3914 | 0.5001 | 3591 | 1595 | 5583 | 5186 |
| 0.50 | 0.8686 | 0.2644 | 0.4054 | 2426 | 367 | 6748 | 2793 |
| 0.75 | 0.9575 | 0.1349 | 0.2366 | 1238 | 55 | 7936 | 1293 |

Primary threshold for all other Phase 5 reports: **0.25**.

- At 0.50: precision is 0.1762 higher and recall is 0.1270 lower than at 0.25 (F1 0.5001 -> 0.4054).
- At 0.75: precision is 0.2650 higher and recall is 0.2565 lower than at 0.25 (F1 0.5001 -> 0.2366).
