# Phase 6 — Focused Error + Forgetting Analysis

Source: measured Phase-5 artifacts (`reports/phase5_nic/`), Scenario NIC / variant inc / run 0, 79 experiences. No retraining was performed; every number below is read from the committed metrics.

## OBSERVED — aggregate effect of replay

| Metric | Naive | Replay | Delta (replay - naive) |
|---|---|---|---|
| Final overall accuracy | 0.0235 | 0.0538 | 0.0303 |
| Final mean forgetting (lower better) | 0.6209 | 0.5364 | -0.0845 |
| Average incremental accuracy | 0.0458 | 0.0632 | 0.0174 |

Definition in effect: forgetting(c, t) = max{acc(c, t') : t' < t} - acc(c, t); the aggregate at experience t is the mean over every class that has a prior measurement. The first experience has no prior measurement (null / N-A). Negative values (a class improved on its previous best) are preserved.

## OBSERVED — highest-forgetting classes (naive)

| Label | Name | Naive forgetting | Naive final acc | Replay final acc |
|---|---|---|---|---|
| 44 | cup5 | 1.0000 | 0.0000 | 0.0000 |
| 33 | ball4 | 0.9933 | 0.0000 | 0.0000 |
| 34 | ball5 | 0.9766 | 0.0000 | 0.0000 |
| 31 | ball2 | 0.9643 | 0.0000 | 0.0000 |
| 40 | cup1 | 0.9133 | 0.0000 | 0.3722 |
| 24 | can5 | 0.9000 | 0.0000 | 0.0000 |
| 48 | remote_control4 | 0.8811 | 0.0000 | 0.3356 |
| 27 | glass3 | 0.8778 | 0.0000 | 0.0000 |
| 46 | remote_control2 | 0.8463 | 0.0000 | 0.0000 |
| 30 | ball1 | 0.8267 | 0.0000 | 0.7167 |

## OBSERVED — lowest final accuracy (naive)

| Label | Name | Naive final acc | Naive forgetting |
|---|---|---|---|
| 1 | plug_adapter2 | 0.0000 | 0.6156 |
| 2 | plug_adapter3 | 0.0000 | 0.6122 |
| 3 | plug_adapter4 | 0.0000 | 0.6389 |
| 4 | plug_adapter5 | 0.0000 | 0.3867 |
| 5 | mobile_phone1 | 0.0000 | 0.3456 |
| 6 | mobile_phone2 | 0.0000 | 0.7156 |
| 7 | mobile_phone3 | 0.0000 | 0.4833 |
| 8 | mobile_phone4 | 0.0000 | 0.7567 |
| 9 | mobile_phone5 | 0.0000 | 0.7256 |
| 11 | scissor2 | 0.0000 | 0.5256 |

## OBSERVED — largest naive/replay differences (final accuracy)

| Label | Name | Naive final | Replay final | Difference |
|---|---|---|---|---|
| 30 | ball1 | 0.0000 | 0.7167 | 0.7167 |
| 15 | light_bulb1 | 0.6547 | 0.1642 | -0.4905 |
| 40 | cup1 | 0.0000 | 0.3722 | 0.3722 |
| 48 | remote_control4 | 0.0000 | 0.3356 | 0.3356 |
| 4 | plug_adapter5 | 0.0000 | 0.3256 | 0.3256 |
| 25 | glass1 | 0.0000 | 0.1947 | 0.1947 |
| 10 | scissor1 | 0.2067 | 0.0433 | -0.1633 |
| 35 | marker1 | 0.0000 | 0.1544 | 0.1544 |
| 12 | scissor3 | 0.0000 | 0.1411 | 0.1411 |
| 20 | can1 | 0.1567 | 0.0500 | -0.1067 |

## OBSERVED — experiences with the largest forgetting increase (naive)

| Experience | Naive forgetting | Naive increase | Replay forgetting | Replay increase |
|---|---|---|---|---|
| 2 | 0.3833 | 0.3509 | 0.3086 | 0.3654 |
| 6 | 0.3942 | 0.0946 | 0.3086 | 0.0511 |
| 13 | 0.4435 | 0.0549 | 0.3952 | 0.0617 |
| 10 | 0.4113 | 0.0463 | 0.3607 | 0.0351 |
| 9 | 0.3649 | 0.0440 | 0.3256 | 0.0501 |
| 12 | 0.3886 | 0.0390 | 0.3336 | 0.0204 |
| 28 | 0.5261 | 0.0358 | 0.4441 | 0.0286 |
| 17 | 0.4308 | 0.0337 | 0.3795 | 0.0078 |

## OBSERVED — representative errors

Candidates evaluated: 100 evaluation images per method; representative examples kept: 12 (errors seen: {'disagreement': 13, 'both_wrong': 83}).

| Image (dataset-relative) | Session | True | Naive prediction | Replay prediction |
|---|---|---|---|---|
| s3/o16/C_03_16_002.png | s3 | light_bulb1 | light_bulb1 (ok, 0.546) | plug_adapter5 (wrong, 0.990) |
| s3/o16/C_03_16_003.png | s3 | light_bulb1 | light_bulb1 (ok, 0.562) | plug_adapter5 (wrong, 0.989) |
| s3/o2/C_03_02_000.png | s3 | plug_adapter2 | light_bulb1 (wrong, 0.712) | plug_adapter5 (wrong, 0.974) |
| s3/o2/C_03_02_001.png | s3 | plug_adapter2 | light_bulb1 (wrong, 0.708) | plug_adapter5 (wrong, 0.975) |
| s7/o31/C_07_31_000.png | s7 | ball1 | light_bulb1 (wrong, 1.000) | ball1 (ok, 0.907) |
| s7/o31/C_07_31_001.png | s7 | ball1 | light_bulb1 (wrong, 1.000) | ball1 (ok, 0.937) |
| s7/o3/C_07_03_000.png | s7 | plug_adapter3 | light_bulb1 (wrong, 1.000) | light_bulb1 (wrong, 1.000) |
| s7/o3/C_07_03_001.png | s7 | plug_adapter3 | light_bulb1 (wrong, 1.000) | light_bulb1 (wrong, 1.000) |
| s10/o4/C_10_04_000.png | s10 | plug_adapter4 | scissor1 (wrong, 0.438) | remote_control1 (wrong, 0.222) |
| s10/o4/C_10_04_001.png | s10 | plug_adapter4 | scissor1 (wrong, 0.431) | remote_control1 (wrong, 0.221) |
| s10/o34/C_10_34_000.png | s10 | ball4 | light_bulb1 (wrong, 0.580) | marker1 (wrong, 0.492) |
| s10/o34/C_10_34_001.png | s10 | ball4 | light_bulb1 (wrong, 0.587) | marker1 (wrong, 0.469) |

## OBSERVED EVIDENCE — sessions/environment

- All evaluated examples come from the fixed official test sessions [3, 7, 10]; those sessions are never used for training (training sessions: [1, 2, 4, 5, 6, 8, 9, 11]).
- Selected example sessions: ['3', '7', '10'].

Per-example observations:

- Session s3 -> light_bulb1 -> session s3 -> class light_bulb1 -> true label 15; naive predicted light_bulb1 (conf 0.546, correct); replay predicted plug_adapter5 (conf 0.990, wrong)
- Session s3 -> light_bulb1 -> session s3 -> class light_bulb1 -> true label 15; naive predicted light_bulb1 (conf 0.562, correct); replay predicted plug_adapter5 (conf 0.989, wrong)
- Session s3 -> plug_adapter2 -> session s3 -> class plug_adapter2 -> true label 1; naive predicted light_bulb1 (conf 0.712, wrong); replay predicted plug_adapter5 (conf 0.974, wrong)
- Session s3 -> plug_adapter2 -> session s3 -> class plug_adapter2 -> true label 1; naive predicted light_bulb1 (conf 0.708, wrong); replay predicted plug_adapter5 (conf 0.975, wrong)
- Session s7 -> ball1 -> session s7 -> class ball1 -> true label 30; naive predicted light_bulb1 (conf 1.000, wrong); replay predicted ball1 (conf 0.907, correct)
- Session s7 -> ball1 -> session s7 -> class ball1 -> true label 30; naive predicted light_bulb1 (conf 1.000, wrong); replay predicted ball1 (conf 0.937, correct)
- Session s7 -> plug_adapter3 -> session s7 -> class plug_adapter3 -> true label 2; naive predicted light_bulb1 (conf 1.000, wrong); replay predicted light_bulb1 (conf 1.000, wrong)
- Session s7 -> plug_adapter3 -> session s7 -> class plug_adapter3 -> true label 2; naive predicted light_bulb1 (conf 1.000, wrong); replay predicted light_bulb1 (conf 1.000, wrong)
- Session s10 -> plug_adapter4 -> session s10 -> class plug_adapter4 -> true label 3; naive predicted scissor1 (conf 0.438, wrong); replay predicted remote_control1 (conf 0.222, wrong)
- Session s10 -> plug_adapter4 -> session s10 -> class plug_adapter4 -> true label 3; naive predicted scissor1 (conf 0.431, wrong); replay predicted remote_control1 (conf 0.221, wrong)
- Session s10 -> ball4 -> session s10 -> class ball4 -> true label 33; naive predicted light_bulb1 (conf 0.580, wrong); replay predicted marker1 (conf 0.492, wrong)
- Session s10 -> ball4 -> session s10 -> class ball4 -> true label 33; naive predicted light_bulb1 (conf 0.587, wrong); replay predicted marker1 (conf 0.469, wrong)

## POSSIBLE FACTORS

- The failure occurred on an example from a different session/environment; the specific causal factor cannot be established from this sample alone.
