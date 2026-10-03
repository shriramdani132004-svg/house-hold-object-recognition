---
title: Continual Household Object Recognition
emoji: 🏠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: mit
short_description: CORe50 50-class continual-learning classifier (name + confidence)
python_version: "3.11"
---

# Continual Household Object Recognition

Phone-first Gradio app for the **frozen** Phase-6 continual-learning model
(Experience Replay, `SmallConvNet`, 50 CORe50 object identities, 64x64
CPU input).

- **Live camera + image upload** — one shared inference callback.
- **Output:** object name, confidence (`0.00%`–`100.00%`), class ID.
- **Classification only:** the selected model has no detector head, so
  **bounding boxes are not supported** and are never drawn.
- Checkpoint `models/continual/final_model.pt` (SHA-256
  `b2f0606cd5e58d811b804f33be48fda779ead5ea1306b1fc951b30af1ca0e351`)
  is loaded once at startup and reused for every frame — never retrained
  in deployment.
- Preprocessing matches Phase-5 training exactly (RGB, bilinear 64x64,
  `/255`, mean/std 0.5).

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
