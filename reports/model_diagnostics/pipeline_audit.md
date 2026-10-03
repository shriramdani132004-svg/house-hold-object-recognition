# Continual-Learning Pipeline Audit (Code-Hardening Phase)

Scope: **code only**. No training run was launched during this phase —
not C2, not C3, not the final run, not any long experiment. The gate
`scripts/verify_training_readiness.py` is the artifact that decides when
training may resume; it reported
`TRAINING READINESS = READY (16/16 checks passed)` at the end of this
work. The frozen baseline (`models/continual/final_model.pt`,
SHA-256 `b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351`)
was re-hashed and unchanged.

## 1. Early stopping: diagnosis and fix (the C1 question)

Completed candidate `c1_corrected_baseline` finished with

| metric | value |
|---|---|
| `dev_best` | 0.3763 |
| `dev_final` | 0.0956 |
| `dev_forgetting` | 0.5838 |
| `avg_incr` | 0.1091 |
| wall clock | 22136 s |

Two independent causes explain the `dev_best` vs `dev_final` gap:

1. **D — missing best-epoch restoration (engineering bug, fixed here).**
   The per-experience early stop was evaluated (12-epoch budget,
   patience 5, seen-class dev subset), but when patience fired the run
   simply stopped on the **last** epoch's weights. Nothing ever
   restored the best-development snapshot, and no global
   `best_model.pt` existed. The checkpoint carried into experience
   *t+1* was therefore whatever happened to be last, not what was
   selected.
   *Fix:* `src/training/early_stop.py` (`EarlyStopping`) now encodes
   the documented rule — a new best is `best + 1e-4`, stop after
   `patience` consecutive non-improvements, non-finite metrics rejected
   loudly — and `ImprovedReplayTrainer` snapshots model weights on every
   improvement, restores them in `_finalize_experience_weights` before
   the state is saved, and atomically persists a global best to
   `best_model.pt` whose threshold is re-loaded on resume. Optimizer
   moments are intentionally *not* rewound (weights only); this is
   stated in the module docstring.
2. **A — genuine forgetting (model behaviour, unchanged).** With best
   epochs restored, a real best-vs-final drop of ~0.58 on the dev
   subset is still expected: sequential fine-tuning erases earlier
   classes. Experience Replay remains the only anti-forgetting method;
   no EWC/LwF/distillation was introduced.

Also fixed in measurement semantics: `dev_best` (max over
per-experience records on the cumulative-classes subset) and
`dev_final` (final experience only) are now documented as
different-comparison metrics in the trainer docstring; they are not
interchangeable.

## 2. Numerical safety

`BaseContinualTrainer._train_epochs` now aborts with
`ContinualTrainingError` **before** `optimizer.step()` when

- the scalar loss is NaN/Inf (message includes experience, epoch, step),
- any parameter gradient is non-finite (message includes the first
  offending parameter names).

A broken run can no longer burn patience, silently write a wrecked
checkpoint, or masquerade as "unlucky epochs". Probe:
`verify_training_readiness.py` check `nonfinite_guards`.

## 3. Preprocessing single source of truth

Previously the resize/normalize pipeline was re-typed ~5 times
(training dataset, tensor cache, live decoding in `improved.py`,
inference preprocessing, EvalCache). Now `src/data/preprocessing.py`
owns `BASE_IMAGE_SIZE`, `INTERPOLATION`, `MEAN/STD`, `decode_image_file`,
`normalize_chw`, `preprocess_pil`; consumers
(`src/training/dataset.py`, `src/training/tensor_cache.py`,
`src/inference/preprocessing.py`, `src/training/improved.py`) import it.
EvalCache stores raw post-resize `uint8` plus shared normalization so
legacy caches remain valid while quantization error disappears. The
verifier enforces this structurally (check `preprocessing_single_source`).

## 4. Class identity and report precedence

- `src/data/class_mapping.py` is the single authoritative 50-class
  mapping: strict validation (count, uniqueness, index coherence,
  `OBJECT_CATEGORY` cross-check) with `ClassMappingError`; consumed by
  `src/evaluation/continual.load_class_names`, the scenario loader, and
  `src.inference.engine.load` (wrapped as `ModelLoadError` at the API
  boundary).
- `scripts/run_phase6_analysis.py::merge_authoritative_class_names`
  wins over report/classification fallbacks at both call sites
  (check `class_names_precedence`).

## 5. Selection-pipeline guards (data leakage)

The dev split is the only selection signal, so it is defended:

- `src/data/continual/validation.py::check_development_split` proves a
  path set is a subset of training references **and** disjoint from the
  held-out s3/s7/s10 evaluation set; wired into
  `scripts/make_dev_split.py` and `scripts/train_candidates.py`.
- A missing manifest aborts with an actionable message
  (`create it first: python scripts/make_dev_split.py`) — no silent
  fallback to held-out data (check `selection_guards`).
- Current manifest `data/splits/nic_inc_run0_dev10_seed42.json`:
  sha256 `3a5dd7dcfbdeb1e332c449ac55d39dac7ae2eebb1b7b3b7ef9b6f589718611c7`,
  107,895 train / 11,999 dev, regenerating it reproduces the same bytes
  (determinism verified by re-running `make_dev_split.py`).

## 6. Checkpoint contract

- Training payloads now record `image_size` and `arch` alongside
  `num_classes` (legacy checkpoints remain valid; readers validate only
  when present).
- `load_final_model` builds the recorded architecture (honouring
  compact candidates), loads `strict=True` with `CheckpointSchemaError`
  on shape mismatch, rejects `image_size` disagreements, and runs
  `assert_model_contract` (structure + forward pass + finite outputs)
  so a wrong model fails at load time, not at predict time.

## 7. Config, logging, dead code

- `ContinualTrainConfig`: unknown keys rejected naming the accepted
  set; `batch_size/learning_rate/image_size/early_stop_patience/
  momentum` bounds validated; dead `tensor_cache_dir` and `from_yaml`
  removed.
- `src/utils/run_logging.py`: deterministic one-line `key=value`
  events (`dev_epoch`, `experience_eval`, `run_start`, lifecycle);
  `train_candidates.py` wires them to `logs/train_candidates.log`
  (per-image events are not logged).
- No-op `_banner`, duplicate `_normalize/_decode_live`, and
  `parse_train_done` removed.

## 8. Metrics

`src/evaluation/metrics.py`: top-k counts, confusion matrix (rows =
true), macro precision/recall/F1 with zero-division mapped to 0.0, and
`average_incremental_accuracy` matching the existing
`run_phase5_experiment` definition. Every function is pure and covered
by hand-computed reference values in `tests/test_metrics.py`.

## 9. End-to-end sample trace (one real record)

Traced with the cached scenario + authoritative mapping:

```
relative_path   s1/o1/C_01_01_000.png
split           train            (never test/s3,s7,s10)
experience_id   0                (first NIC experience)
label           0 -> class name "plug_adapter1" (class_mapping.py)
in dev exp0?    False            (belongs to the 107,895-train pool)
images_root     <project>/data/raw/core50/dataset/core50_128x128   (resolved at runtime)
cache key       "s1/o1/C_01_01_000.png"      (TensorCache, raw post-resize uint8)
```

Journey: filelist line (`loader.py`, `SampleRecord` with split/line
number/object/session provenance) → scenario experience record →
train/dev partition (`make_dev_split.py`, seeded per class/experience,
leakage-checked) → `train_candidates.py` filters dev paths out of the
training samples → `TensorCache` decodes once via
`src.data.preprocessing.decode_image_file` (BILINEAR to 64 px) →
`AugmentedCachedDataset` serves `normalize_chw` tensors (training adds
only flip/shift/brightness) → `BaseContinualTrainer._train_epochs`
with NaN/Inf guards around every optimizer step → `ImprovedReplayTrainer`
dev evaluation each epoch via `EarlyStopping` → best weights restored,
`state.json` + `best_model.pt` written → per-experience record with
`evaluate_dataset` counts → `compute_forgetting` /
`average_incremental_accuracy` in the candidate summary. Held-out
evaluation (s3/s7/s10) is touched only by the separate evaluation path
and never by the selection signal.

## 10. Verification evidence

- `python scripts/verify_training_readiness.py` →
  **TRAINING READINESS = READY (16/16 checks passed)**
- `python -m pytest tests -q` → all tests green (see final run in the
  session log; includes new suites: `test_metrics.py`,
  `test_early_stop.py` with one-batch smoke + tiny-overfit tests)
- `python -m compileall src scripts tests` → clean
- Baseline SHA-256 re-verified unchanged by check `baseline_checkpoint`
- Secrets/absolute-path sweep passes (checks `no_secrets`,
  `no_absolute_paths`)

## 11. Explicitly NOT done (by instruction)

- No C2/C3/final training launches, no held-out s3/s7/s10 evaluation,
  no hyperparameter search, no long experiments.
- Experience Replay remains the only anti-forgetting method.
- Historical results and the Phase-5 baseline were not modified.
- `phases map.txt` contains a pre-existing, unrelated rewrite that is
  deliberately left uncommitted.
