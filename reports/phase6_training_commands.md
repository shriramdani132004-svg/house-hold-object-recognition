# Phase 6 - Commands Executed

All commands run from the project root with the project virtual environment.

## Preflight and schedule measurement (temporary probes, not committed)

```powershell
# hardware probe: torch/cuda/xpu availability, versions (inline python -c checks)
# dataset pre-state counts (inline python -c checks): 40890 / 4544 / 1965 images, 214983 instances
.\.venv\Scripts\python.exe scripts\benchmark_training_speed.py --tag c --threads 16 --workers 0 --batch 16
.\.venv\Scripts\python.exe scripts\benchmark_training_speed.py --tag a --threads 16 --workers 4 --batch 16
.\.venv\Scripts\python.exe scripts\benchmark_training_speed.py --tag d --threads 16 --batch 16 --channels-last
.\.venv\Scripts\python.exe scripts\benchmark_training_speed.py --tag e --threads 16 --batch 16 --compile
# (a fraction=0.05 probe first established 9.5 s/iteration -> 6.9 h/epoch)
```

## Training

```powershell
.\.venv\Scripts\python.exe scripts\train_model.py --dry-run
# launched 2026-10-02 05:15 local as a detached process (PID 23744), console log redirected:
#   models\training\train_console.log  /  models\training\train_err.log
.\.venv\Scripts\python.exe scripts\train_model.py
# completed 2026-10-02 18:03 local: TRAIN_DONE status=completed, epochs_completed=2,
# wall_s=46086.5, best.pt=true, last.pt=true
```

## Post-train validation and reports

```powershell
.\.venv\Scripts\python.exe scripts\validate_training.py
```

## Verification gates

```powershell
.\.venv\Scripts\python.exe -m compileall src scripts tests
.\.venv\Scripts\python.exe -m pytest tests -q
git status --porcelain
```

## Notes

- The fraction/threads/workers/channels_last/compile benchmarks were run in a temporary
  directory to fix the epoch schedule before launch; `scripts/benchmark_training_speed.py`
  is the committed, re-runnable form of the same measurement.
- No GPU/XPU was available; no environment packages were installed or upgraded in this phase.
