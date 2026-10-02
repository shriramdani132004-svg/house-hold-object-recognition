# CORe50 Dataset

Primary dataset for the continual-recognition assignment, acquired and
verified during Phase 2. All information below is taken from the official
CORe50 materials or measured from the acquired files on this machine —
nothing is estimated.

## Official Source

- Project page: <https://vlomonaco.github.io/core50/>
- Dataset + benchmark description, download links, benchmark scenarios:
  the *Dataset*, *Benchmark* and *Download* sections of that page.
- Dataset paper: Vincenzo Lomonaco and Davide Maltoni, *"CORe50: a new
  Dataset and Benchmark for continual Object Recognition"*, Proceedings of
  the 1st Annual Conference on Robot Learning (CoRL), PMLR 78:17-26, 2017
  (<https://arxiv.org/abs/1705.03550>).
- Second paper (NICv2 / fine-grained scenarios): Vincenzo Lomonaco, Davide
  Maltoni and Lorenzo Pellegrini, *"Fine-Grained Continual Learning"*,
  arXiv:1907.03799, 2019 (<https://arxiv.org/abs/1907.03799>).

## Repository

- Official repository: <https://github.com/vlomonaco/core50>
- A snapshot of the official repository (`master` branch) is stored at
  `metadata/core50-official/` and provides:
  - the official Python data loader (`scripts/python/data_loader.py`),
  - the official label→name helper (`scripts/python/from_labels_to_names.py`),
  - the official experiment configurations for the three scenarios
    (`confs/sI` = NI, `confs/sII` = NC, `confs/sIII` = NIC),
  - the official `extras/` bundle (complete `paths.pkl`, `labels.pkl`,
    `LUP.pkl`, `core50_labels.txt`, NICv2 filelists, official result tables).
- The official setup script `scripts/bash/fetch_data_and_setup.sh` downloads
  exactly one dataset archive — `core50_128x128.zip` — from
  `http://bias.csr.unibo.it/maltoni/download/core50/`. Phase 2 follows that
  documented method (see *Acquisition Method*).

## Dataset Purpose

CORe50 — **CO**ntinual **O**bject **Re**cognition, 50 objects — is the
benchmark dataset for continual / lifelong object recognition. It was
designed so that the *same objects* reappear in *different sessions* with
changing background, lighting, pose, viewpoint and hand occlusion, which is
what makes session-based (continual) learning experiments meaningful. It
supports:

- continual object recognition (classification) at object-identity level
  (50 classes) or category level (10 classes) — the object-identity level is
  the default and harder task;
- object detection (bounding boxes are published for the full-size frames);
- approximate object segmentation (via the depth modality, not used here).

This project uses CORe50 for the assignment: continual recognition of
household objects across sessions (NI / NC / NIC scenarios).

## Object Identities

50 physical objects, directory ids `o1` … `o50`. Official names are stored
in the official repository file `metadata/core50-official/extras/
core50_labels.txt` and re-emitted per object (with counts) by the validator
into `metadata/object_mapping.json`:

| Ids | Names | Category |
|---|---|---|
| o1–o5 | `plug_adapter1` … `plug_adapter5` | plug adapter |
| o6–o10 | `mobile_phone1` … `mobile_phone5` | mobile phone |
| o11–o15 | `scissor1` … `scissor5` | scissors |
| o16–o20 | `light_bulb1` … `light_bulb5` | light bulb |
| o21–o25 | `can1` … `can5` | can |
| o26–o30 | `glass1` … `glass5` | glass |
| o31–o35 | `ball1` … `ball5` | ball |
| o36–o40 | `marker1` … `marker5` | marker |
| o41–o45 | `cup1` … `cup5` | cup |
| o46–o50 | `remote_control1` … `remote_control5` | remote control |

Both levels are preserved: **50 object identities** *and* **10 categories**
(`metadata/object_mapping.json`). Nothing is collapsed to 10 classes.

## Categories

The 10 official categories (project page, object→category block):

```
plug adapters, mobile phones, scissors, light bulbs, cans,
glasses, balls, markers, cups, remote controls
```

The validator stores them in official order as
`metadata/object_mapping.json → category_order`, using singular directory
style (`plug adapter`, `mobile phone`, …) for machine-readable keys.

## Sessions

- **11 sessions** `s1` … `s11` (8 indoor + 3 outdoor per the official page),
  each recorded as 15-second videos at 20 fps with a Kinect 2.0 sensor.
- Every session contains all 50 objects: `sN/oM/C_NN_MM_FFF.png`.
- Official train/test convention: sessions **s3, s7, s10 are the fixed test
  sessions**; the remaining **8 sessions (s1, s2, s4, s5, s6, s8, s9, s11)
  are training sessions**.
- Measured image counts per session (all match the official
  `Color128x128.tsv` table exactly):

  | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 14,989 | 14,986 | 14,992 | 14,995 | 14,966 | 14,989 | 14,994 | 14,984 | 14,994 | 14,986 | 14,991 |

- Total: **164,866 images** (official figure, verified on disk).

## RGB / Other Modalities Used

- **Used: the official 128x128 RGB (color) frames** — the exact archive the
  official `fetch_data_and_setup.sh` downloads and the archive the official
  benchmark data loader consumes.
- Frame naming: `C_[session]_[object]_[frame].png` (C = color), e.g.
  `C_05_33_179.png`. The `D_*.png` depth frames live in separate official
  archives.
- **Not downloaded (optional, documented on the official page):**
  - `core50_350x350.zip` (full-size RGB, 27.7 GB) — the 128x128 set is the
    benchmark recognition data; not needed for continual recognition;
  - `core50_128x128_depth.zip` / `core50_350x350_depth.zip` (depth /
    segmentation modality);
  - `core50_imgs.npz` (packed copy of the same 128x128 images);
  - TensorFlow object-detection records and the cluttered-detection test set.
- Bounding-box metadata **was** acquired (`metadata/bbox/`, 550 files = 11
  sessions x 50 objects). It is stored for later detection work; the boxes
  are expressed relative to the *full-size 350x350* frames, as documented on
  the official page.

## Dataset Layout

```
data/raw/core50/
├── downloads/                     # verbatim official archives (kept until re-validation)
│   ├── core50_128x128.zip         # 5,892,103,007 B — official image archive
│   ├── batches_filelists.zip      #    45,418,248 B — official NI/NC/NIC filelists
│   ├── dataset_dims.zip           #         2,578 B — official per object/session counts
│   ├── bbox.zip                   #     1,390,289 B — official bounding boxes
│   ├── core50-master.zip          #    41,886,216 B — official repository snapshot
│   ├── core50_class_names.txt, labels2names.pkl, labels.pkl, LUP.pkl
│   └── MANIFEST.json              # URLs, byte sizes, SHA-256 of every download
├── dataset/
│   └── core50_128x128/            # OFFICIAL session structure (never flattened)
│       ├── s1/ … s11/
│       │   └── o1/ … o50/
│       │       └── C_NN_MM_FFF.png     # 164,866 PNGs, 128x128 RGB
├── filelists/                     # official scenario filelists (verbatim)
│   ├── NI_inc/  NI_cum/           # New Instances  (8 batches + test, 10 runs)
│   ├── NC_inc/  NC_cum/           # New Classes    (9 batches + test)
│   ├── NIC_inc/ NIC_cum/          # New Instances + Classes (79 batches + test)
│   ├── NIC_v2_79/ NIC_v2_196/ NIC_v2_391/   # NICv2 (fine-grained) variants
│   └── …/runN/{train_batch_XX_filelist.txt, test_filelist.txt}
├── metadata/
│   ├── object_mapping.json        # 50 objects: id, official name, category, counts
│   ├── dataset_dims/              # official Color128x128.tsv + siblings
│   ├── bbox/s1…s11/CropC_oNN.txt  # official bounding boxes (550 files)
│   ├── core50-official/           # official repository snapshot (reference)
│   └── …                          # labels.pkl, LUP.pkl, paths.pkl (complete copies)
├── DATASET_INFO.md                # this file
└── README.md                      # short layout guide
```

## Continual Scenarios

Official resources for the three benchmark scenarios are present and
validated (scenario definitions from the official project page):

| Scenario | Meaning | Official resource | Observed |
|---|---|---|---|
| **NI** — New Instances | new views/conditions of *known* classes | `filelists/NI_inc`, `filelists/NI_cum` | 10 runs each, 8 train batches + test per run |
| **NC** — New Classes | batches introduce *new* classes | `filelists/NC_inc`, `filelists/NC_cum` | 10 / 5 runs, 9 train batches + test per run |
| **NIC** — New Instances + Classes | both at once | `filelists/NIC_inc`, `filelists/NIC_cum` | 10 / 3 runs, 79 train batches + test per run |

- Official experiment configuration files (Sacred/JSON + Caffe prototxt) are
  in `metadata/core50-official/confs/` (`sI` = NI, `sII` = NC, `sIII` = NIC).
- NICv2 (the "Fine-Grained Continual Learning" variant with 79 / 196 / 391
  experiences) filelists are in `filelists/NIC_v2_*`.
- The cumulative variants published by the official archive contain fewer
  runs than the 10 described in the docs: `NC_cum` ships **5** runs and
  `NIC_cum` ships **3** runs. This is an official-archive fact (recorded in
  `reports/phase2_core50_summary.json → filelists.*.runs_match_documentation`),
  not a local defect.
- Phase 2 only *locates and validates* these resources; no continual
  experiment is run in this phase.

## Filelists

- Source: official `batches_filelists.zip` (byte-identical to the copy in
  the official repository `extras/`) plus `batches_filelists_NICv2.zip` from
  the official repository snapshot.
- Format: one `relative/path.png label` pair per line, paths relative to
  `dataset/core50_128x128/` (e.g. `s11/o1/C_11_01_000.png 0`), labels are
  integers. The test filelist of every run contains the same fixed test set
  (verified by hashing).
- Scale: **8,060 filelist files, 33,895,788 lines**, referencing
  **164,866 unique image paths** — every single one resolves to a real file
  on disk (0 unresolved).
- Note from the official page: for NC the object→label mapping differs per
  run (labels are made contiguous per incremental batch by design).

## Acquisition Method

Scripts (run from the project root, project venv only):

```bash
.\.venv\Scripts\python.exe scripts\download_core50.py   # resumable download + SHA-256 manifest
.\.venv\Scripts\python.exe scripts\extract_core50.py    # structure-preserving extraction
.\.venv\Scripts\python.exe scripts\validate_core50.py   # full validation + summary + samples
```

- Only official hosts are used:
  - images + full-size archive: `http://bias.csr.unibo.it/maltoni/download/core50/`
    (hosted by the University of Bologna, the CORe50 authors' institution);
  - small benchmark files: `https://vlomonaco.github.io/core50/data/…`
    (the official project page);
  - repository snapshot: `https://github.com/vlomonaco/core50`.
- Downloads are resumable (HTTP `Range`), retry with backoff, and verified
  against the byte sizes reported by the official hosts; every acquired file
  is recorded with its **SHA-256** in `downloads/MANIFEST.json`.
- Downloaded on 2026-10-02 (`acquired_utc: 2026-10-02T15:50:13+00:00`),
  total 6,005,983,106 bytes (5.59 GiB) across 9 resources.
- Extraction preserves the official directory names, session ids, object ids
  and frame names — nothing is flattened, renamed or reordered.

### Known issue in the official static hosting (handled)

Four small files served from `vlomonaco.github.io/core50/data/` are
**truncated server-side** (their published byte size already ends mid-file):
`core50_class_names.txt` (183 B, stops at `scissor`), `labels2names.pkl`,
`labels.pkl` and `LUP.pkl`. Phase 2 therefore uses the **byte-complete
copies shipped inside the official repository** (`extras/core50_labels.txt`,
`extras/labels.pkl`, `extras/LUP.pkl`, `extras/paths.pkl`), which load
correctly. The truncated downloads are kept untouched in `downloads/` as
evidence; the validator never depends on them. `batches_filelists.zip` and
`dataset_dims.zip` from the page are byte-identical to the repository copies
(verified by MD5).

## License / Usage

- The official repository `LICENSE` and the official project page both state:
  **Creative Commons Attribution 4.0 International (CC BY 4.0)** —
  <https://creativecommons.org/licenses/by/4.0/>.
- Practical consequences for this project: attribution is required (cite the
  two CORe50 papers above), raw images must not be redistributed as our own
  work, and results must credit CORe50 / University of Bologna.
- The dataset page asks users to cite the dataset papers when using the data
  or any of the released resources.

## Reproducibility

- Exact byte sizes and SHA-256 digests of every downloaded artifact:
  `downloads/MANIFEST.json`.
- Official reference counts: `metadata/dataset_dims/Color128x128.tsv`
  (per object x session and per session totals) — matched exactly by the
  validator (550/550 cells, 0 mismatches).
- Official path order: `metadata/core50-official/extras/paths.pkl`
  (164,866 entries) — set-equal to the files on disk.
- Official per-run label/look-up tables: `extras/labels.pkl`, `extras/LUP.pkl`.
- Full machine-readable results of this phase:
  `reports/phase2_core50_summary.json` (status `PASS`).
- Nothing in the dataset directories is Git-tracked, so the manifest plus
  these references are what make the acquisition reproducible.

## Storage Requirements

| Component | Bytes | GiB |
|---|---|---|
| `downloads/` (official archives + manifest) | 6,005,987,672 | 5.59 |
| `dataset/` (164,866 PNGs) | 5,858,196,330 | 5.46 |
| `filelists/` (8,060 text filelists) | 872,705,756 | 0.81 |
| `metadata/` (dims, bbox, official repo snapshot) | 151,196,369 | 0.14 |
| **Total under `data/raw/core50/`** | **12,888,086,127** | **12.00** |

Free disk space before/after this phase: ~255 GiB / ~248 GiB (C:).
The archives in `downloads/` are retained (they cost 5.59 GiB) so the
extracted tree can always be re-verified against them.

## Integrity Verification

`scripts/validate_core50.py` (full run, status **PASS**, measured
2026-10-02):

- structure: 11 sessions, 550 object directories, 164,866 files,
  0 zero-byte files, 0 illegal names, 0 non-PNG files;
- official dims cross-check: 550 object/session cells, **0 mismatches**;
- PNG signature check: **164,866 / 164,866 valid**, 0 failures;
- representative decode (Pillow): 43 images spanning **11 sessions,
  38 objects, 10 categories**, 0 failures, all 128x128;
- official `paths.pkl` vs disk: **identical** (164,866 = 164,866);
- filelists: 9 scenario directories, 33,895,788 lines, **0 unresolved
  paths**, 0 malformed lines, fixed test set identical across runs;
- bounding boxes: 550 / 550 files parse;
- human-inspection sample: 16 images in `reports/phase2_core50_samples/`
  covering all 11 sessions, all 10 categories, 16 distinct objects.

## Notes

- Session structure is **load-bearing** for the assignment: later phases must
  consume `s1…s11` (or the official filelists) and must not flatten CORe50
  into a random train/test split.
- The COCO 2017 material under `data/raw/coco/` is the earlier prototype of
  this project and was left untouched by Phase 2.
- Temp download/extraction logs are written to `logs/` (Git-ignored).
