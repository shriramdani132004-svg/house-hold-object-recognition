# `data/raw/core50/` — CORe50 raw dataset (Phase 2)

Everything in `data/raw/core50/` is **acquired from the official CORe50
sources** and is **Git-ignored**; this documentation is tracked under
`docs/dataset/core50/`. Full
documentation, licence and verification results: [DATASET_INFO.md](DATASET_INFO.md).

```
downloads/   official archives + MANIFEST.json (SHA-256 of every file)
dataset/     core50_128x128/s1..s11/o1..o50/C_NN_MM_FFF.png  (164,866 PNGs)
filelists/   official NI / NC / NIC (+ NICv2) scenario filelists, verbatim
metadata/    object_mapping.json, official dims tables, bbox files,
             official repository snapshot (data loader, confs, extras)
```

Rebuild / verify from the project root:

```bash
.\.venv\Scripts\python.exe scripts\download_core50.py   # resumable
.\.venv\Scripts\python.exe scripts\extract_core50.py    # resumable
.\.venv\Scripts\python.exe scripts\validate_core50.py   # full validation
```

Do **not** flatten or rename `s*/o*` directories — the session structure is
required by the continual-learning phases.
