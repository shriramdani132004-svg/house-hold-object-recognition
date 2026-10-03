PHASE 6 — ERROR ANALYSIS + FINAL MODEL SELECTION

Dataset:
CORe50

Scenario:
NIC

Variant:
inc

Run:
0

Experiences:
79

------------------------------------------------------------
NAIVE
------------------------------------------------------------

Final accuracy:
0.0234590411811794

Final forgetting:
0.620896

Average incremental accuracy:
0.045834

------------------------------------------------------------
REPLAY
------------------------------------------------------------

Final accuracy:
0.05378902428177533

Final forgetting:
0.536408

Average incremental accuracy:
0.063249

------------------------------------------------------------
OBSERVED FORGETTING
------------------------------------------------------------

- Highest forgetting classes (naive): cup5 (1.0000), ball4 (0.9933), ball5 (0.9766), ball2 (0.9643), cup1 (0.9133)
- Lowest-accuracy classes (naive): plug_adapter2 (0.0000), plug_adapter3 (0.0000), plug_adapter4 (0.0000), plug_adapter5 (0.0000), mobile_phone1 (0.0000)
- Experiences with largest forgetting increase (naive): 2, 6, 13, 10, 9
- Naive vs Replay differences: replay final accuracy 0.0538 vs naive 0.0235; replay final forgetting 0.5364 vs naive 0.6209; replay average incremental accuracy 0.0632 vs naive 0.0458

------------------------------------------------------------
REPRESENTATIVE ERRORS
------------------------------------------------------------

- 12 examples (candidates evaluated: 100 per method; errors: naive 87, replay 92)
- Sessions covered: s3, s7, s10
- s3/o16/C_03_16_002.png (s3): true=light_bulb1; naive=light_bulb1 (0.546, ok); replay=plug_adapter5 (0.990, wrong)
- s3/o16/C_03_16_003.png (s3): true=light_bulb1; naive=light_bulb1 (0.562, ok); replay=plug_adapter5 (0.989, wrong)
- s3/o2/C_03_02_000.png (s3): true=plug_adapter2; naive=light_bulb1 (0.712, wrong); replay=plug_adapter5 (0.974, wrong)
- s3/o2/C_03_02_001.png (s3): true=plug_adapter2; naive=light_bulb1 (0.708, wrong); replay=plug_adapter5 (0.975, wrong)
- s7/o31/C_07_31_000.png (s7): true=ball1; naive=light_bulb1 (1.000, wrong); replay=ball1 (0.907, ok)
- s7/o31/C_07_31_001.png (s7): true=ball1; naive=light_bulb1 (1.000, wrong); replay=ball1 (0.937, ok)
- s7/o3/C_07_03_000.png (s7): true=plug_adapter3; naive=light_bulb1 (1.000, wrong); replay=light_bulb1 (1.000, wrong)
- s7/o3/C_07_03_001.png (s7): true=plug_adapter3; naive=light_bulb1 (1.000, wrong); replay=light_bulb1 (1.000, wrong)
- s10/o4/C_10_04_000.png (s10): true=plug_adapter4; naive=scissor1 (0.438, wrong); replay=remote_control1 (0.222, wrong)
- s10/o4/C_10_04_001.png (s10): true=plug_adapter4; naive=scissor1 (0.431, wrong); replay=remote_control1 (0.221, wrong)
- s10/o34/C_10_34_000.png (s10): true=ball4; naive=light_bulb1 (0.580, wrong); replay=marker1 (0.492, wrong)
- s10/o34/C_10_34_001.png (s10): true=ball4; naive=light_bulb1 (0.587, wrong); replay=marker1 (0.469, wrong)

- Evidence-based observations: all examples are from official test
  sessions [3, 7, 10] that are never used for
  training; session-level environmental factors vary per the dataset
  documentation, but no specific causal factor is claimed from these
  samples alone.

------------------------------------------------------------
FINAL MODEL
------------------------------------------------------------

Selected method:
replay

Checkpoint:
models/continual/final_model.pt

Reason:
Evidence-based comparison of Phase-5 measured accuracy, forgetting,
old-knowledge retention, and incremental performance. Selection rule:
Replay is at least as good as naive on every primary measured criterion and strictly better on at least one; select replay. Measured detail: final_accuracy: naive 0.0235 vs replay 0.0538 (replay better); final_forgetting: naive 0.6209 vs replay 0.5364 (replay better); old_knowledge_retention: naive 0.0235 vs replay 0.0538 (replay better); average_incremental_accuracy: naive 0.0458 vs replay 0.0632 (replay better).

SHA-256:
b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351
