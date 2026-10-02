"""Method A — Naive Continual Learning (Phase 4).

Sequential training without replay: one persistent model is trained on
experience 0, then continued (never re-initialized) through every later
experience in official order, saving model/optimizer/scheduler state after
each one. Future experiences are never read; the official evaluation set is
never used for training.
"""

from __future__ import annotations

from src.training.base import BaseContinualTrainer


class NaiveContinualTrainer(BaseContinualTrainer):
    """Naive sequential trainer: same model, no replay memory.

    Semantics (enforced by :class:`BaseContinualTrainer`):

    - ``initialize(scenario)`` creates the model, optimizer and initial
      checkpoint exactly once;
    - ``train_experience(state, experience)`` reloads the previous
      checkpoint from disk, trains only the current experience's official
      training samples, and saves the updated state;
    - experience 0 first, then strictly ``i + 1`` — official order, no
      resets, no future data.
    """

    method_name = "naive"
