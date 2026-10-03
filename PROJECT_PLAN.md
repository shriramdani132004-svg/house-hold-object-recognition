# PROJECT_PLAN.md — Household Object Recognition Roadmap

Complete high-level roadmap for the project. Each phase is completed and
verified before moving to the next.

The **assignment** is continual recognition of household objects with the
CORe50 dataset (see `phases map.txt`, the canonical phase list). The table
below is the original static-detection roadmap; its Phases 1–6 were delivered
as the COCO/YOLO **prototype** and remain intact.

| # | Phase | Goal | Key Outputs |
|---|-------|------|-------------|
| 1 | **Setup** | Project foundation: structure, venv, Git, docs, basic deps | Folder tree, `README.md`, `AGENTS.md`, `requirements.txt`, `.gitignore`, `LICENSE`, initial commit |
| 2 | **Dataset** | Select and acquire a suitable public household-object detection dataset | Downloaded raw data in `data/raw/`, dataset + license documentation |
| 3 | **Preparation** | Clean data, select classes, convert annotations, create splits, validate | `data/processed/`, `data/splits/`, validation report |
| 4 | **Analysis** | Class distribution, sample annotated images, object statistics, data problems | Figures in `reports/figures/`, analysis notes |
| 5 | **Baseline** | Load a pretrained lightweight YOLO model and test zero-shot / before training | Baseline metrics record |
| 6 | **Training** | Train the custom detector; save checkpoints, logs, metrics | Trained weights in `models/`, training logs |
| 7 | **Evaluation** | Precision, recall, mAP50, mAP50-95, confusion matrix, inference speed | Evaluation report + plots in `reports/` |
| 8 | **Improvement** | Error analysis: misses, false positives, wrong classes, bad boxes → fix data/training | Error analysis report, improvement actions |
| 9 | **Final Model** | Run improved training, compare experiments, select and freeze final model | Frozen final model version + comparison table |
| 10 | **Inference** | Clean Python inference: image prediction, boxes, scores, object summaries | `src/inference/` API + CLI |
| 11 | **Web App** | Gradio interface: upload image → household-object detections | `app/` Gradio application |
| 12 | **Testing** | Automated tests + edge-case testing of the full inference application | `tests/` suite, CI-green |
| 13 | **GitHub** | Clean repo, professional README, examples, docs, `.gitignore`, `LICENSE`, GitHub Actions | Polished public repository |
| 14 | **Deployment** | Deploy Gradio app to Hugging Face Spaces, incl. model storage and config | Live Space + `app.py` deployment config |
| 15 | **Final Polish** | Full start-to-finish verification, fix broken paths, remove cruft, portfolio-ready repo | Verified, tidy repository |
| 16 | **Documentation** | Final project report: dataset, architecture, training, evaluation, error analysis, deployment, limitations, future work | `reports/` final report |

### Prototype status (COCO/YOLO)

- [x] Phase 1 — Setup
- [x] Phase 2 — Dataset (COCO 2017 — prototype)
- [x] Phase 3 — Preparation
- [x] Phase 4 — Analysis
- [x] Phase 5 — Baseline
- [x] Phase 6 — Training
- [ ] Phase 7 — Evaluation (prototype; paused — superseded by the continual assignment)
- [ ] Phase 8 — Improvement
- [ ] Phase 9 — Final Model
- [ ] Phase 10 — Inference
- [ ] Phase 11 — Web App
- [ ] Phase 12 — Testing
- [ ] Phase 13 — GitHub
- [ ] Phase 14 — Deployment
- [ ] Phase 15 — Final Polish
- [ ] Phase 16 — Documentation

---

## Continual Recognition Roadmap (canonical: `phases map.txt`)

Assignment phases for **continual recognition of household objects with
CORe50**:

| # | Phase | Goal | Status |
|---|-------|------|--------|
| 1 | **Setup** | Project foundation: structure, venv, Git, docs, tests | [x] Complete |
| 2 | **CORe50 Dataset** | Acquire + verify the official CORe50 dataset, session structure, object mapping, NI/NC/NIC resources, documentation | [x] Complete (2026-10-02) |
| 3 | **Continual Data Pipeline** | Session/experience loading, sequential experiences, official filelists, no leakage | [x] Complete (2026-10-02) |
| 4 | **Naive + Experience Replay** | Sequential baseline and the one required anti-forgetting method | [x] Complete (2026-10-02) |
| 5 | **Main NIC Experiment + Evaluation** | Naive vs Replay with accuracy / forgetting measurements | [x] Complete (2026-10-03) |
| 6 | **Error Analysis + Final Model** | Focused forgetting/error analysis, freeze the selected checkpoint | [x] Complete (2026-10-03) |
| 7 | **Inference + Phone Web App** | Inference API + Gradio live-camera app | [ ] |
| 8 | **Testing + Deployment** | Practical tests + Hugging Face Spaces deployment | [ ] |
| 9 | **GitHub + Documentation** | Polished repo + final report | [ ] |
| 10 | **Final Verification + Demonstration** | End-to-end checks, phone demo | [ ] |

Phase 2 deliverables recorded: `docs/dataset/core50/DATASET_INFO.md`,
`DATASET.md` (CORe50 primary section), `reports/phase2_core50_summary.json`,
`reports/phase2_core50_samples/`, `scripts/download_core50.py`,
`scripts/extract_core50.py`, `scripts/validate_core50.py`,
`tests/test_phase2_core50.py`.

### PHASE 3 — CONTINUAL DATA PIPELINE

STATUS: IMPLEMENTED (2026-10-02)

- **Scenario-based data loading**: `load_scenario(scenario, variant, run)`
  in `src/data/continual/` provides SCENARIO → EXPERIENCES → train +
  evaluation samples for later training phases
  (`for experience in scenario.experiences: ...`).
- **Official filelist usage**: scenarios are read from the official
  `data/raw/core50/filelists/{NI,NC,NIC}_{inc,cum}/run{N}/` tree (plus the
  official `NIC_v2_{79,196,391}` variants); no splits are invented.
- **Sequential experiences**: official batch order preserved, experience 0
  first, no shuffling, no future experience visible to an earlier one.
- **Train/evaluation separation**: the fixed official test set
  (sessions s3/s7/s10, 44,972 samples) is disjoint from all training data.
- **Leakage safeguards** (`src/data/continual/validation.py`): train/eval
  overlap, future-data leakage (disjoint incremental batches; monotone
  cumulative chains), experience ordering, duplicate references, unresolved
  image paths, unknown object/session/class ids, label ranges.
- **Reproducibility**: explicit scenario/variant/run selection, deterministic
  loading (repeat loads compare equal), cached re-parsing, and only
  project-relative paths in persisted metadata.
- **NI/NC/NIC support**: all 9 official variants are discovered and
  validated; official NC per-run label remapping is reported as a fact
  (`label_equals_object_minus_one: false`).
- **Primary future scenario**: NIC (Phase 4+ experiments).
- **Artifacts**: `reports/phase3_continual_pipeline_manifest.json`,
  `scripts/inspect_core50_scenarios.py`, `scripts/build_phase3_manifest.py`,
  `configs/continual.yaml`, `tests/test_phase3_continual_pipeline.py`.

Phase 3 does **not** train models, implement replay, or compute accuracy —
those belong to later phases.

### PHASE 4 — NAIVE + EXPERIENCE REPLAY

STATUS: IMPLEMENTED (2026-10-02)

- **Method A — Naive continual learning** (`NaiveContinualTrainer`,
  `src/training/naive.py`): the model is created exactly once, then the
  SAME model continues through every official experience — no per-
  experience reset, no fresh initialization, official order enforced
  (experience 0 first, then strictly `i + 1`), only the current
  experience's official training samples are read.
- **Method B — Experience Replay** (`ReplayContinualTrainer` +
  `ReplayMemory`, `src/training/replay.py`): the single anti-forgetting
  method. Per experience: current samples + replay samples from earlier
  experiences → combined into each training batch → train → update memory →
  persist state. The memory starts empty (no replay at experience 0).
- **Persistent model state**: `ContinualTrainingState` with
  `save_state()` / `load_state()` / `validate_state()` /
  `checkpoint_exists()` (`src/training/state.py`) persist model,
  optimizer, scheduler and replay memory to `state.json` + `checkpoint.pt`
  under `models/continual/<scenario>_<variant>_run<id>/<method>/`
  (git-ignored, outside the dataset tree, project-relative metadata only —
  no absolute personal paths).
- **Bounded replay memory**: fixed capacity with FIFO eviction over
  lightweight `SampleRecord` references (image paths are re-read from the
  official tree; images are never copied). Evaluation/test samples,
  future-experience samples and out-of-order additions are rejected.
- **Deterministic replay sampling**: seeded uniform sampling
  (`ReplayMemory.sample(k, seed=...)`), seeded model initialization and
  seeded data-loader shuffling.
- **Shared trainer interface**: `build_continual_trainer("naive" |
  "replay", config)` — Phase 5 switches methods without touching training
  code; unknown method names fail with a clear `ValueError`.
- **Configuration**: generic, explicit defaults in `configs/continual.yaml`
  (`continual` / `training` / `replay` sections) — final Phase-5
  experiment settings are deliberately NOT defined yet.
- **Model**: compact PyTorch `SmallConvNet` classifier
  (`src/training/model.py`) for CORe50 object-identity classification;
  the Ultralytics/YOLO stack remains exclusive to the untouched COCO
  detection prototype.
- **Tests**: `tests/test_phase4_continual_training.py` — 24 focused,
  fast tests on tiny synthetic fixtures plus one official NIC
  experience-0 smoke step (two optimizer steps, no full experiment).

### PHASE 5 — MAIN NIC CONTINUAL EXPERIMENT + EVALUATION

STATUS: COMPLETE (2026-10-03)

- **Experiment**: Scenario NIC variant `inc` run 0 — 79 official
  experiences (119,894 training references). Both methods trained
  independently from ONE shared initial model state (checksum verified at
  every initialize/resume), identical architecture and hyperparameters
  (SmallConvNet width 32, 64x64 input, 3 epochs, batch 64, lr 1e-3 Adam,
  seed 42); the ONLY difference is the replay section (FIFO capacity 2000,
  replay batch 16).
- **Evaluation**: fixed official test set (sessions 3/7/10, 44,972
  samples, one shared uint8 cache) evaluated after every experience for
  both methods; per-experience overall / old / new / full / per-class
  accuracy, forgetting (documented definition) and average incremental
  accuracy recorded.
- **Measured results** (`reports/phase5_nic/`): final overall accuracy
  naive 0.0235 vs replay 0.0538; final mean forgetting naive 0.6209 vs
  replay 0.5364; average incremental accuracy naive 0.0458 vs replay
  0.0632 — replay better on all three measured metrics (single run,
  single seed).
- **Artifacts**: `configs/phase5_nic.yaml`,
  `scripts/run_phase5_experiment.py`, `src/evaluation/continual.py`,
  `src/utils/experiment_display.py`, `tests/test_phase5_metrics.py`;
  outputs under `reports/phase5_nic/` (metrics JSONs, `comparison.csv`,
  `per_class_metrics.json`, `experiment_summary.json`,
  `phase5_nic_report.md`, 4 plots) and git-ignored
  `models/continual/phase5_nic/{shared,naive,replay}/`.
- **Integrity**: all 8 checks PASS (official experience order, no future
  experience leakage, no evaluation samples in training, same initial
  state both methods, independent runs with identical settings, bounded
  replay memory, no image duplication, reproducible configuration).
- **Verification**: full suite 212 tests passing; interrupt-safe resume
  proven by a real mid-run interruption and recovery.

Numbering note: `phases map.txt` describes the two methods as its Phases 4
(naive) and 5 (replay); assignment Phase 4 implements both together, and
its Phase 6 "Continual Experiment" plus Phase 7 "Continual Evaluation"
correspond to assignment Phase 5.

### PHASE 6 — ERROR ANALYSIS + FINAL MODEL SELECTION

STATUS: COMPLETE (2026-10-03)

- **Analysis** (`reports/phase6_analysis/`): deterministic re-analysis of
  the committed Phase-5 metrics — `forgetting_analysis.json`,
  `class_analysis.json`, `experience_analysis.json` and
  `phase6_error_analysis.md`. Key measured findings: highest naive
  forgetting cup5 1.0000, ball4 0.9933, ball5 0.9766; lowest naive final
  accuracy plug_adapter2–5 and mobile_phone1 at 0.0000; largest naive
  forgetting increases after experiences 2, 6, 13, 10, 9; replay's
  measured effect +0.0303 final accuracy, −0.0845 final forgetting,
  +0.0174 average incremental accuracy.
- **Targeted representative inference**: 100 deterministic test-split
  candidates (8 hardest classes, round-robin across sessions 3/7/10)
  classified by both frozen final checkpoints; 12 session-balanced
  representative errors (disagreement-first ranking, per-class cap 2)
  recorded in `representative_errors.json`, example PNGs under
  `reports/phase6_analysis/examples/` (git-ignored).
- **Environment analysis**: `environment_analysis.json` strictly
  separates OBSERVED evidence from POSSIBLE factors using only the
  documented session facts; no causal claim is made without evidence
  (fallback uncertainty sentence everywhere).
- **Final model selection** (`final_model_selection.json/.md`): four
  primary measured criteria (final accuracy, final forgetting,
  old-knowledge retention, average incremental accuracy) — replay is at
  least as good on every criterion and strictly better on all four, so
  REPLAY is selected by the documented evidence-driven rule.
- **Frozen checkpoint**: `models/continual/final_model.pt` — byte copy of
  the Phase-5 replay checkpoint, SHA-256 verified identical to source,
  portable metadata in `models/continual/final_model.json` (relative
  paths only), confirmed git-ignored.
- **Integrity**: all attestations PASS/PRESERVED — no retraining (Phase-5
  state/checkpoint SHA-256s byte-identical before/after), no dataset
  modification (164,866-file non-decoding count), no future-data access,
  no evaluation data used for training, no image duplication, portable
  metadata (0 absolute paths in generated reports), COCO prototype
  preserved.
- **Artifacts**: `src/evaluation/error_analysis.py`,
  `scripts/run_phase6_analysis.py` (7-step driver with overall progress
  display), `tests/test_phase6_analysis.py` (20 focused tests); full
  suite 232 tests passing.
- **Numbering note**: `phases map.txt` Phases 8 (focused
  error/forgetting analysis) and 9 (final model selection) together are
  assignment Phase 6.

## Notes

- Dependencies are installed incrementally: only what the current phase
  needs. Heavy ML dependencies (e.g. `ultralytics`, `torch`) are added when
  training phases begin, not during setup.
- Dataset license and model weights licensing must be checked and recorded
  in `README.md` / dataset docs during Phase 2.
- The COCO prototype under `data/raw/coco/`, `data/processed/`,
  `models/` and `reports/` is preserved as-is; CORe50 lives exclusively
  under `data/raw/core50/`.
