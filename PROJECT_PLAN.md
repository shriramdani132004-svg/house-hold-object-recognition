# PROJECT_PLAN.md — Household Object Recognition Roadmap

Complete high-level roadmap for the project. Each phase is completed and
verified before moving to the next.

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

## Status

- [x] Phase 1 — Setup
- [x] Phase 2 — Dataset
- [x] Phase 3 — Preparation
- [ ] Phase 4 — Analysis
- [ ] Phase 5 — Baseline
- [ ] Phase 6 — Training
- [ ] Phase 7 — Evaluation
- [ ] Phase 8 — Improvement
- [ ] Phase 9 — Final Model
- [ ] Phase 10 — Inference
- [ ] Phase 11 — Web App
- [ ] Phase 12 — Testing
- [ ] Phase 13 — GitHub
- [ ] Phase 14 — Deployment
- [ ] Phase 15 — Final Polish
- [ ] Phase 16 — Documentation

## Notes

- Dependencies are installed incrementally: only what the current phase
  needs. Heavy ML dependencies (e.g. `ultralytics`, `torch`) are added when
  training phases begin, not during setup.
- Dataset license and model weights licensing must be checked and recorded
  in `README.md` / dataset docs during Phase 2.
