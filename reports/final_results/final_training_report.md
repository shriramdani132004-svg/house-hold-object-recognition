# ONE-SHOT FINAL CORe50 Training and Held-Out Evaluation Report

Protocol date: 2026-10-04 (UTC+05:30). Scenario: CORe50 NIC, incremental
variant, run 0 (79 experiences, 50 classes). Anti-forgetting method:
**Experience Replay only** (no EWC/LwF/distillation). The official
held-out sessions (s3/s7/s10) were read **once**, after training and
freezing were complete, by `scripts/evaluate_final.py` — never for
training, selection, debugging, or tuning.

## 1. Final run

| Item | Value |
|---|---|
| Command | `python scripts/train_candidates.py --final --config configs/final_training.yaml` |
| Config | `configs/final_training.yaml` (fingerprint `06f3d864d7809cdb`) |
| Model | CompactResNet (GroupNorm residual), width 32, 635,602 params, 64x64 input |
| Optimizer | AdamW lr 1e-3, wd 1e-4, label smoothing 0.1, scheduler none |
| Replay | reservoir, capacity 10,000, replay batch 32 (ratio 32:64) |
| Augmentation | flip, rotation ±3°, translation, brightness/contrast, gaussian noise |
| Epoch budget | 12 max per experience, early-stop patience 5 (development subset) |
| Seed / threads | 42 / torch_threads 8 (pinned, measured fastest on this CPU) |
| Training data | 107,895 references (development 11,999 excluded from training) |
| Completed | 79/79 experiences, 910 epochs, 20,226 optimizer steps |
| Wall clock | 66,677.6 s (18 h 31 min), CPU only |
| Candidate dir | `models/continual/candidates/final_full_nic/` |

Development-set selection used the per-experience **full** development
accuracy records (identical set and definition across experiences). Their
argmax is experience 78 — the final experience — so the trainer's final
`checkpoint.pt` is the development-selected model (asserted by the freeze
script, not assumed). The trainer's separate global *subset* best
(`best_model.pt`, subset value 0.972) was **excluded** because subset
accuracies across experiences cover different numbers of seen classes and
are not comparable; the reason is recorded in the frozen payload's
`selection.excluded_signal` block.

## 2. Development results (selection metric, legal for selection)

| Metric | Final run | C1 reference |
|---|---|---|
| Dev final accuracy (full, cumulative) | **0.9041** | 0.0956 |
| Dev best accuracy | 0.9041 | 0.3763 |
| Dev forgetting (final) | **0.0320** | 0.5838 |
| Dev average incremental accuracy | **0.5624** | 0.1091 |

## 3. Frozen artifact (Phase 40)

| Item | Value |
|---|---|
| Checkpoint | `models/continual/final_model.pt` |
| SHA-256 | `0db904f44c726ae1e1f894d1c7f8f81afe638350aa0ba5bfefb95efc8c84d1bf` |
| Source | `models/continual/candidates/final_full_nic/checkpoint.pt` (sha `89297b67…`) |
| Metadata | `models/continual/final_model.json` (phase 7) |
| Historical baseline | preserved byte-identical at `models/continual/baseline_replay_phase5.pt`, SHA-256 `b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351` (pinned before and after the freeze) |
| Label mapping | `core50-object-mapping-v1`, sha `5d28556368a38f161160773a0210bdad6a9d73ba839d6e110066421104facad2` |

## 4. Held-out evaluation (ONE session, 44,972 samples, s3/s7/s10)

Report: `reports/final_results/final_evaluation.json`.

| Metric | Final | Phase-5 baseline | Delta |
|---|---|---|---|
| Top-1 accuracy | **0.3517** | 0.0538 | **+0.2979** |
| Top-5 accuracy | **0.6008** | 0.1842 | **+0.4165** |
| Macro F1 | **0.3512** | 0.0251 | **+0.3261** |
| Macro precision | 0.4192 | 0.0219 | +0.3973 |
| Macro recall | 0.3517 | 0.0538 | +0.2979 |
| Old classes (45) accuracy | 0.3752 | 0.0419 | **+0.3333** |
| New classes (5) accuracy | 0.1404 | 0.1604 | −0.0200 |

Definitions: *new* = the classes introduced by the most recent
introducing experience (experience 28: `plug_adapter2`,
`plug_adapter5`, `scissor3`, `can5`, `remote_control4`); the final
experiences introduce no new classes. *Old* = the other 45 classes.
Per-session top-1:

| Session | Final | Baseline | n |
|---|---|---|---|
| s3 | 0.5056 | 0.0831 | 14,992 |
| s7 | 0.2842 | 0.0322 | 14,994 |
| s10 | 0.2652 | 0.0460 | 14,986 |

Best classes (top-1): `ball4` 0.901, `ball1` 0.897, `marker1` 0.818,
`marker4` 0.783, `marker2` 0.768. Weakest classes (top-1):
`plug_adapter4` 0.000, `cup2` 0.002, `plug_adapter2` 0.003,
`plug_adapter5` 0.008, `plug_adapter3` 0.028 — the visually
near-identical plug-adapter family remains the dominant confusion under
unseen sessions, which also explains the small new-class regression (two
of the five new classes are plug adapters).

## 5. Honest limitations

- The development (0.9041) and held-out (0.3517) figures measure
  different things: the development split comes from the *training*
  sessions, while s3/s7/s10 are entirely unseen capture sessions. The
  gap quantifies novel-session generalization; it is not a label or
  pipeline error (label audit: 0 failures over 164,866 references).
- Only one training run and one held-out evaluation were performed, as
  required; no hyperparameter search was conducted, so no claim of a
  global optimum is made.
- The first invocation of the evaluation script crashed after the final
  model's forward pass (legacy baseline payload lacked the `image_size`
  key); the same evaluation session was completed by a rerun with no
  training, tuning, or selection in between (`session_note` in the JSON
  report).
- Training ran on CPU only (Intel Core Ultra 5 125H, torch 2.14.1
  CPU build).

## 6. Reproduction commands

```text
python scripts/validate_label_references.py --check-files   # label gate
python scripts/prove_labels.py                              # identity proof
python scripts/train_candidates.py --final --config configs/final_training.yaml
python scripts/freeze_final_model.py                        # Phase 40 freeze
python scripts/evaluate_final.py                            # ONE held-out eval
python scripts/smoke_inference.py                           # inference smoke
python scripts/verify_app.py                                # app verification
python scripts/verify_training_readiness.py                 # 16-check gate
python -m pytest tests -q                                   # full suite
```
