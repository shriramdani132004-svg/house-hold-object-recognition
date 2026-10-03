# Baseline Preservation Record (Phase-5 Experience Replay)

Recorded by Step 0 of the model-improvement work package. The frozen
baseline model is preserved and must not be overwritten.

## Model

- Baseline model path: `models/continual/final_model.pt`
- Preservation copy: `models/continual/baseline_replay_phase5.pt`
- SHA-256 (both, verified identical): `b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351`
- Architecture: `SmallConvNet` (`small_cnn`), width 32, 100,146 parameters
- Input: 64x64 RGB, 50-way linear head

## Preprocessing (training = evaluation = inference)

RGB conversion -> bilinear resize to 64x64 -> CHW float32 -> `/255.0` ->
normalize with mean/std `(0.5, 0.5, 0.5)` (value range `[-1, 1]`).

## Training configuration (configs/phase5_nic.yaml)

Scenario NIC / variant `inc` / run 0 / seed 42 / 79 experiences.
epochs 3, batch 64, optimizer Adam, lr 0.001, weight decay 0.0,
scheduler none, device cpu.

## Replay configuration

Experience Replay (the only anti-forgetting method): FIFO memory,
capacity 2000 references, replay batch 16 concatenated with each
current batch of 64, replay seed 42.

## Stored baseline metrics (reports/phase5_nic/)

- Final replay accuracy: `0.05378902428177533` (5.38%)
- Final naive accuracy: `0.0234590411811794`
- Replay forgetting: `0.536408` (naive `0.620896`)
- Replay average incremental accuracy: `0.063249` (naive `0.045834`)
- Classes with zero final accuracy: 37 / 50
- Wall clock (both methods): 5729.938 s
