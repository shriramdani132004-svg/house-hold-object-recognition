# Training-Readiness Gate

`scripts/verify_training_readiness.py` is the blocking gate for this
project: **no training of any kind may be launched until it passes.**

## Run it

```
python scripts/verify_training_readiness.py
```

- exit code `0` — all 16 checks passed; the last line reads
  `TRAINING READINESS = READY (16/16 checks passed)`
- exit code `1` — at least one check failed; the last lines read
  `TRAINING READINESS = NOT READY (n/16 checks failed)` followed by the
  failed check names. Training must not start.

The script performs **no training**: it imports modules, runs tiny
in-memory probes, reads metadata manifests, and hashes files. Typical
runtime is under a minute (the cached NIC scenario metadata load
dominates).

## The 16 checks

| # | Check | What it proves |
|---|---|---|
| 1 | `imports` | core `src` modules import without error |
| 2 | `class_mapping` | the authoritative 50-class mapping loads, unique names, labels exactly 0..49 |
| 3 | `no_absolute_paths` | no machine-specific paths hardcoded in `src/`, `app/`, `configs/`, `scripts/` |
| 4 | `no_secrets` | no credential-like literals in code/configs |
| 5 | `config_validation` | invalid training configs are rejected naming the offending key |
| 6 | `preprocessing_single_source` | exactly one preprocessing implementation; training/inference/cache/eval all import it |
| 7 | `dev_manifest` | the train-only dev manifest exists and is internally consistent (79 experiences, recorded evaluation sessions `[3, 7, 10]`) |
| 8 | `dev_leakage` | every one of the 11,999 dev references is a training reference and disjoint from held-out s3/s7/s10 |
| 9 | `selection_guards` | `train_candidates.py` aborts loudly on a missing manifest and calls the leakage check |
| 10 | `early_stop_wiring` | best-epoch restore hooks are overridden in `ImprovedReplayTrainer`; patience/restore semantics verified |
| 11 | `nonfinite_guards` | NaN/Inf losses and gradients abort training before `optimizer.step()` |
| 12 | `baseline_checkpoint` | the frozen baseline `models/continual/final_model.pt` matches its recorded SHA-256, passes schema validation, loads, and satisfies the model contract |
| 13 | `metrics_spot_check` | metric functions reproduce hand-computed reference values |
| 14 | `structured_logging` | deterministic one-line `key=value` event formatting |
| 15 | `dead_code_removed` | known dead helpers stay removed |
| 16 | `class_names_precedence` | authoritative mapping wins over report/classification fallbacks |

A crashed check counts as a failed check.

## When to re-run

Before every training launch, after any change to training code, data
splits, configs, or checkpoint loading, and before the final commit of a
hardening phase. Keep the READY output next to the run's command log as
evidence the gate was respected.
