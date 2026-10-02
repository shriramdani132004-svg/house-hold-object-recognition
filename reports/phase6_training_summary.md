# Phase 6 - Training Summary

- **Status:** COMPLETE - 2 / 2 epochs, one controlled experiment, seed 42, device CPU (XPU unavailable).
- **Model:** COCO-pretrained `yolo26n` fine-tuned on 40,890 train images; checkpoint `models/training/household_yolo26n/weights/best.pt`.
- **Training wall:** 12.80 h (5.69 h + 6.93 h per epoch incl. validation).
- **Label:** VALIDATION RESULT (validation split 4,544 images / 21,100 instances; test split untouched).

## Headline metrics (validation, best.pt)

| Metric | Value | Zero-shot same split | Change |
|---|---|---|---|
| Precision | 62.11% | 73.67% | -11.55 pts |
| Recall | 50.07% | 43.81% | +6.27 pts |
| mAP50 | 54.06% | 60.55% | -6.49 pts |
| mAP50-95 | 37.78% | 44.44% | -6.66 pts |

## Key facts

- 2 epochs x ~6.9 h/epoch measured on CPU; schedule fixed before launch; 50-epoch cap and patience 15 configured.
- Dataset integrity: MATCH vs phase-start counts (raw 123298 files / 41,368,946,252 bytes).
- Reports: `reports/phase6_training_report.md`, `reports/phase6_training_results.json`, `reports/phase6_training_commands.md`.
- Next: Phase 7 may use the test split for the first time.
