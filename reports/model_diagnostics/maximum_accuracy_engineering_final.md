# Maximum Accuracy Engineering — Final Record (Label-Safe CORe50)

Date: 2026-10-04. Scope: everything that was diagnosed, hardened, measured,
or decided while turning the label-broken Phase-5 pipeline into the
one-shot final run. Companion report:
`reports/final_results/final_training_report.md` (results).

## 1. Label forensics (Phases 0–13) — the hard blocker, resolved first

Symptom: Phase-5 baseline top-1 on held-out sessions was 0.0538 — near
chance — and official class names disagreed with the locally stored
labels.

Findings and fixes:

- `class_mapping.py` now carries `MAPPING_VERSION =
  "core50-object-mapping-v1"`, `mapping_checksum()` (sha
  `5d28556368a38f161160773a0210bdad6a9d73ba839d6e110066421104facad2`)
  and `assert_matches_official_names()`.
- Hard rules enforced in `src/data/continual/scenarios.py` and
  `loader.py`: official label rule applies to **NIC and NI only**; the
  NC_inc protocol's single-context exemption is documented, never
  silently widened.
- `src/data/continual/label_audit.py` +
  `scripts/validate_label_references.py`: full sweep of **164,866
  references, 0 failures**, `label == object_id - 1` everywhere outside
  the documented exemption; `--check-files` additionally opens every
  referenced image.
- `scripts/prove_labels.py`: identity proof for all **50/50** class ids.
- `tests/test_label_integrity.py`: 11 tests pinning the rule; both archs
  (`small_cnn`, `compact_resnet`) are covered elsewhere by
  `tests/test_model_contract.py`.

Hard rule held: any mapping disagreement aborts — it is never papered
over with a compatibility shim.

## 2. Training / checkpoint / inference hardening (Phases 14–33)

- `config.py`: `label_smoothing` (0.1), `torch_threads` validated
  [1, 64] and applied via `torch.set_num_threads` (thrashing-free),
  `cache_dir` passthrough.
- `base.py`: NaN/Inf loss aborts; checkpoint payload carries
  `training_config`, `config_fingerprint`, `mapping_version`,
  `mapping_checksum`, `epochs_completed`, `steps_completed`; guarded
  mapping keys (file-exists before load); `StateCompatibilityError` on
  arch / num_classes / mapping mismatch instead of a silent partial load.
- `checkpoint_schema.py`: optional keys documented, `SUPPORTED_ARCHS`
  validation, on-disk `mapping_checksum` comparison, arch-aware
  `load_final_model()`.
- `improved.py` augmentation: flip + rotation ±3° (affine_grid/grid_sample)
  + translation + brightness/contrast + σ0.02 gaussian noise.
- Replay decode fixed: `dataset.py` exposes a uint8 tensor cache and
  `replay.py` consumes it, removing per-replay-step PNG decoding.
- Driver (`train_candidates.py`): `--final --config <yaml>` mode with
  mutual-exclusion and exit-2 validation, development data **always**
  excluded from training, development evaluation/records/best-checkpoint
  **always** recorded (so selection evidence exists for the final run).

## 3. Measured performance decisions (CPU-only, no guesses)

Environment: Intel Core Ultra 5 125H, 16 GB RAM, torch 2.14.1 CPU.

| Measurement | Result | Decision |
|---|---|---|
| Train, CompactResNet w32 @64px, b128, th8 | ~35 img/s | final architecture + resolution |
| Train, SmallConvNet @64px, b64, th8 | ~118 img/s | too weak for accuracy target |
| Inference, CompactResNet @64px, b256, th8 | ~107 img/s | app/eval path |
| Threads = 14 (all cores) | slower than th8 (thread thrashing) | **pin `torch_threads: 8`** |
| bf16 autocast | **hangs** on this build | rejected |
| 96 px input | ~2.2x slower end-to-end | rejected for 64 px |
| Replay PNG decode per step | 146–379 img/s bottleneck | replaced by uint8 tensor cache |
| CompactResNet vs SmallConvNet | 635,602 vs 100,597 params | accuracy-first choice |

Honest correction: the 64 px CompactResNet run was projected at ~9–10 h
from b128 micro-benchmarks; the actual run (b64, patience-5 early stop,
full development evaluation per experience) took **18 h 31 min**
(66,677.6 s, 910 epochs, 20,226 steps).

## 4. Selection-bias discovery (why `best_model.pt` was rejected)

`best_model.pt` (mtime 05:27) was the trainer's *global* development-subset
best, reached during early experiences when few classes existed — its
0.972 value comes from an **easier, smaller class subset** than later
experiences, so it is not comparable across experiences.

Evidence used instead (all inside the frozen payload):

1. The per-experience **full** development records share one definition
   (identical sample set per experience).
2. Their argmax is experience **78** — the final experience — so the
   final-state `checkpoint.pt` *is* the development-selected model.
3. `scripts/freeze_final_model.py` asserts `argmax(records) == last
   record == state.current_experience == 78` and the trainer's
   `dev_final == dev_best == 0.9041`; a mismatch aborts the freeze.

The rejected signal and its reason are preserved in the payload's
`selection.excluded_signal` block (audit trail, not deleted evidence).

## 5. Freeze provenance (Phase 40)

- `scripts/freeze_final_model.py` pins the historical baseline sha
  (`b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351`)
  before and after the write, copies the selected final-state
  checkpoint atomically, and records `provenance` (source checkpoint +
  sha `89297b67613897e60cc2b3170654ff2311b704d14eacea27cde5245066ad6e82`,
  baseline sha, frozen UTC), `selection` (dev argmax proof,
  `held_out_sessions_used: false`), config fingerprint `06f3d864d7809cdb`,
  mapping version/checksum.
- Final artifact sha: `0db904f44c726ae1e1f894d1c7f8f81afe638350aa0ba5bfefb95efc8c84d1bf`.
- Baseline preserved byte-identical at
  `models/continual/baseline_replay_phase5.pt` (verified by
  `tests/test_repair_integrity.py::test_final_model_metadata_and_hash`).
- Model binaries remain git-ignored (freeze script runs
  `git check-ignore` and aborts if the file would be tracked).

## 6. Phase-6/7 coexistence fix

`tests/test_phase6_analysis.py::test_phase6_driver_runs_end_to_end`
re-executes `scripts/run_phase6_analysis.py`, whose historical step 6
re-exported `final_model.pt` — clobbering the frozen phase-7 artifact.
Fixed in `run_phase6_analysis.py`: step 6 detects an existing payload
carrying `provenance`/`selection` keys and **preserves** it (logging that
the phase-6 selection is history-only), and its load test became
arch-aware (reads `payload["arch"]`). Hash-integrity tests
(`test_repair_integrity`, `test_phase7_inference`) now accept either the
locked phase-6 state or the phase-7 frozen state with the baseline
pinned. Verified: full suite runs with the frozen sha unchanged.

## 7. Gate status at hand-off (Phases 34 + 43–46)

| Gate | Status |
|---|---|
| `python -m compileall src scripts tests app.py app` | OK |
| `python -m pytest tests -q` | **481 passed** |
| Label audit (`validate_label_references --check-files`) | 0 failures |
| Label identity proof (`prove_labels.py`) | 50/50 |
| `scripts/verify_training_readiness.py` | READY 16/16 |
| `scripts/smoke_inference.py` | OK (5 predictions) |
| `scripts/verify_app.py` | OK (HTTP 200, compact_resnet @64px) |
| Frozen sha after full suite | unchanged (`0db904f4…`) |

## 8. Remaining risks (documented, not hidden)

- Novel-session shift: dev 0.9041 vs held-out 0.3517 — the plug-adapter
  family collapses under unseen sessions (`plug_adapter4` 0.000).
- New-class delta −0.0200: two of the five new classes are plug
  adapters, sharing the same confusion.
- Single run, single evaluation, no sweeps (protocol constraint).
- One evaluation-session crash/resume recorded in
  `reports/final_results/final_evaluation.json` (`session_note`).
