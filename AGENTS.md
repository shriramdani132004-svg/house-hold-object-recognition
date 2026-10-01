# AGENTS.md — Instructions for AI Coding Sessions

## Project Purpose

Household Object Recognition is a computer-vision project that detects and
labels common household objects in images. The project covers the full
workflow: dataset acquisition, data preparation, analysis, training an
object-detection model (YOLO family), evaluation, inference, a Gradio web
app, testing, and deployment to Hugging Face Spaces.

Work is organized in phases (see `PROJECT_PLAN.md` and `phases map.txt`).
**Only perform the phase that has been requested.** Do not start dataset
collection, training, or deployment work during setup tasks.

## Python Coding Conventions

- Python 3.14 (matches the local interpreter). Use modern, readable code.
- Use `src/` as the importable package root (`src.data`, `src.training`,
  `src.evaluation`, `src.inference`, `src.utils`).
- Add `if __name__ == "__main__":` guards to runnable scripts in `scripts/`.
- Prefer `pathlib.Path` over `os.path` for all file paths.
- Type hints on function signatures are encouraged.
- Keep functions small and focused; no dead code, no commented-out blocks.
- Follow PEP 8. Run `python -m compileall src scripts tests` after edits.
- Configuration goes in `configs/` (YAML), never inline magic numbers in
  logic code.
- Third-party dependencies are declared in `requirements.txt` only.

## Folder Responsibilities

| Path | Responsibility |
|---|---|
| `app/` | Gradio / web application entry point |
| `configs/` | YAML config files (training, data, inference) |
| `data/raw/` | Original downloaded dataset (never edited by hand) |
| `data/processed/` | Cleaned/converted dataset artifacts |
| `data/splits/` | Train/val/test split files |
| `examples/input/` | Sample input images for demos and tests |
| `examples/output/` | Sample detection outputs (generated) |
| `models/` | Trained model weights and checkpoints |
| `notebooks/` | Exploratory / analysis notebooks |
| `reports/` | Evaluation reports; `reports/figures/` for plots |
| `scripts/` | Runnable pipeline scripts (download, prepare, train…) |
| `src/` | Reusable library code (`data`, `training`, `evaluation`, `inference`, `utils`) |
| `tests/` | Automated tests (`pytest`) |
| `.github/workflows/` | CI workflow definitions |

Large artifacts (`data/`, `models/`, `examples/output/`) are git-ignored
except for small placeholder/readme files.

## Testing Expectations

- Tests live in `tests/` and are written with `pytest`.
- Run the full suite before declaring any change complete:
  `python -m pytest tests -q`
- If tests do not yet exist for a new feature, add at least one basic test.
- **Every change must be verified by running the appropriate checks** — at
  minimum `python -m compileall src scripts tests`, plus `pytest` when
  tests are affected.

## Hard Rules

1. **No hardcoded absolute paths.** Never write `C:\Users\...` or
   `/home/...` in code or configs. Resolve paths relative to the project
   root (e.g. via `Path(__file__).resolve().parents[n]`) or use paths
   relative to the working directory. Keep the project portable.
2. **Never commit secrets.** No API keys, tokens, passwords, or HF/GitHub
   credentials in code, configs, notebooks, or commits. Secrets belong in
   environment variables or a git-ignored `.env` file only.
3. **Verify changes.** After modifying code, run the relevant tests /
   compile checks listed above and confirm they pass before finishing.
4. **Stay in scope.** Only execute the phase requested by the user; do not
   begin later phases unprompted.
5. **Respect `phases map.txt`** as the canonical phase list for the project.

## Git Conventions

- Commit messages: concise, imperative mood, e.g.
  `Add data split generation script`.
- Stage only intended files; never commit `data/`, `models/`, virtual
  environments, or generated outputs.
