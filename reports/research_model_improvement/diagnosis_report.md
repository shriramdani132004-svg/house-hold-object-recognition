# Model Diagnosis Report (Step 1)

Read-only audit of the existing continual-learning pipeline before any
training. Evidence sources: source code audit, stored Phase-5 metrics
(`reports/phase5_nic/`), scenario structure, and a FIFO memory-coverage
simulation. No training was performed during diagnosis.

## A. Label pipeline — PASS

- Official `object_mapping.json`: 50 objects, `label = object_id - 1`,
  one-to-one, stable across train/eval/inference (`load_class_names`
  uses the same rule).
- Scenario metadata: `label_min=0`, `label_max=49`,
  `label_equals_object_minus_one=true`.
- Class ordering is identical in training, evaluation cache, and the
  deployed 50-class mapping. No mismatch found.

## B. Data pipeline — PASS

- Official CORe50 filelists; NIC `inc` run 0; 79 experiences in official
  order; 119,894 training references; 44,972 evaluation references from
  sessions s3/s7/s10 only.
- Train/evaluation separation, future-experience protection, duplicate
  checks, and leakage checks are enforced by `validate_scenario` and by
  the existing test suite (all passing).
- No evaluation sample appears in training or in the replay memory.

## C. Preprocessing — PASS (consistent everywhere)

Training (`ContinualImageDataset`), evaluation (`EvalCache` round-trip),
and inference (`src/inference/preprocessing.py`) all perform: RGB
conversion -> bilinear resize -> CHW float -> `/255` -> mean/std 0.5.
No train/eval/inference mismatch.

## D. Training loop — no bug found; duration is low

- Adam lr 1e-3, weight decay 0, no scheduler, cross-entropy, batch 64,
  3 epochs per experience, model mode and optimizer state handled
  correctly; checkpoint continuation reloads model/optimizer/scheduler
  every experience (`base.py`).
- Diagnosis finding: only **3 epochs per experience** (explicitly
  documented as an under-fitting compromise in `configs/phase5_nic.yaml`).

## E. Experience Replay — PRIMARY ROOT CAUSE

Stored replay metrics (`train.accuracy_mean`) show the model learns each
current experience well (0.46–0.96 train accuracy), yet final accuracy is
5.38% with 37/50 classes at **zero** accuracy and forgetting 0.536.

Structural analysis of the memory:

- Every experience contains exactly 5 classes (~1,500 samples); every
  class appears in exactly 8 experiences (once per training session).
- FIFO eviction with capacity 2000 keeps only ~1.33 experiences.
- Simulated FIFO coverage over the official ordering:

  | capacity | classes covered at experience 78 |
  |---|---|
  | 2000 (baseline) | **7 / 50** |
  | 5000 | 15 / 50 |
  | 10000 | 29 / 50 |

- Replay batch 16 vs current 64 (20% rehearsal) further weakens the
  gradient from old classes.

Consequence: at any moment the buffer rehearses only the most recent
classes; every older class is driven to zero accuracy. This — not
preprocessing, labels, or leakage — explains the 5.38% result.

## F. Model — underpowered but not broken

SmallConvNet width 32 = 100,146 parameters. It memorizes current
experiences (train accuracy high) but has weak transfer (train accuracy
0.59 vs evaluation new-class accuracy 0.35 at experience 10).
BatchNorm running statistics additionally drift to the most recent
5 classes, mis-normalizing older classes at evaluation time.

## G. Training signal — learning happens, retention fails

- Replay per-experience train loss/accuracy: learning is healthy within
  each experience (final-experience train accuracy 0.85).
- Old-class evaluation accuracy collapses to 0.02 by experience 10 and
  stays near zero; new-class accuracy at experience 10 is 0.355.
- Forgetting (existing project definition): 0.0 -> 0.536 monotonically.
- The model is learning and then catastrophically forgetting; it is not
  merely under-trained.

## Verdict

No implementation bug in labels, data separation, preprocessing, or
checkpoint continuation. The low accuracy is caused by a configuration
and memory-policy weakness in Experience Replay (FIFO starvation of old
classes) combined with a weak rehearsal ratio, short training duration,
lack of augmentation, and a small model. These are exactly the candidate
dimensions explored in Steps 3–6.

## Machine constraints

18 CPUs, CPU-only torch, 15.4 GB RAM total (~3.3 GB available during
diagnosis), 264 GB free disk. Higher resolutions require memory-mapped
decoded caches rather than full in-RAM datasets.
