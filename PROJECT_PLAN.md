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
| 3 | **Continual Data Pipeline** | Session/experience loading, sequential experiences, official filelists, no leakage | [ ] |
| 4 | **Naive + Experience Replay** | Sequential baseline and the one required anti-forgetting method | [ ] |
| 5 | **Main NIC Experiment + Evaluation** | Naive vs Replay with accuracy / forgetting measurements | [ ] |
| 6 | **Error Analysis + Final Model** | Focused forgetting/error analysis, freeze the selected checkpoint | [ ] |
| 7 | **Inference + Phone Web App** | Inference API + Gradio live-camera app | [ ] |
| 8 | **Testing + Deployment** | Practical tests + Hugging Face Spaces deployment | [ ] |
| 9 | **GitHub + Documentation** | Polished repo + final report | [ ] |
| 10 | **Final Verification + Demonstration** | End-to-end checks, phone demo | [ ] |

Phase 2 deliverables recorded: `data/raw/core50/DATASET_INFO.md`,
`DATASET.md` (CORe50 primary section), `reports/phase2_core50_summary.json`,
`reports/phase2_core50_samples/`, `scripts/download_core50.py`,
`scripts/extract_core50.py`, `scripts/validate_core50.py`,
`tests/test_phase2_core50.py`.

## Notes

- Dependencies are installed incrementally: only what the current phase
  needs. Heavy ML dependencies (e.g. `ultralytics`, `torch`) are added when
  training phases begin, not during setup.
- Dataset license and model weights licensing must be checked and recorded
  in `README.md` / dataset docs during Phase 2.
- The COCO prototype under `data/raw/coco/`, `data/processed/`,
  `models/` and `reports/` is preserved as-is; CORe50 lives exclusively
  under `data/raw/core50/`.
