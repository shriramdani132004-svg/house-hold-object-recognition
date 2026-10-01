# Phase 5 — Reproduction Commands

Run from the project root. Use the project virtual environment:

```powershell
$env:Path = ".venv\Scripts";$env:Path
```

## Single-image baseline inference

```powershell
python scripts\baseline_inference.py --source data\processed\household_objects\images\test\000000000139.jpg --save
python scripts\baseline_inference.py --source path\to\image.jpg --conf 0.5 --save
```

Prints one line per detection (confidence, class, pixel box), the
inference time, and with `--save` stores the annotated copy under
`examples/output/`.

## Full test-split evaluation (Phase 5 report suite)

```powershell
python scripts\evaluate_baseline.py --config configs/baseline.yaml
```

- Model: `models/baseline/yolo26n.pt` (auto-downloaded when absent)
- Inference: CPU, imgsz=640, conf floor 0.01, one pass,
  then all thresholds computed offline.
- Runtime: roughly 10-15 minutes on a laptop CPU (about 0.4 s per
  image including decode); progress is shown as a tqdm bar with
  count, percent, elapsed and ETA.
- The detections cache `reports/baseline_detections.json` is written
  after a full pass; reruns reuse it and skip inference.

Force a fresh pass:

```powershell
python scripts\evaluate_baseline.py --recompute
```

## Outputs

```text
reports/baseline_report.md
reports/baseline_results.json
reports/baseline_confidence_analysis.md
reports/baseline_error_examples.md
reports/baseline_samples/
reports/baseline_commands.md
```

## Tests

```powershell
python -m pytest tests -q
```
