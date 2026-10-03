"""Explicit, testable early-stopping with a best-epoch selection rule.

The rule implemented here is deliberately simple and documented:

- ``observe(value, epoch)`` is called once per training epoch with the
  validation metric (higher is better);
- a new best is any value exceeding ``best_value + min_delta``; the
  caller is expected to snapshot the model weights at that moment;
- training stops when ``patience`` consecutive observations failed to
  improve;
- ``should_restore`` reports whether a best snapshot exists and must be
  restored when the experience ends, so the state carried to the next
  experience is the BEST development epoch, not the last one.

This module contains no model or data dependencies so the stopping
decision itself can be unit-tested against synthetic metric sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    """Best-value/patience tracker for one training experience."""

    patience: int
    min_delta: float = 1e-4
    best_value: float = field(default=float("-inf"))
    best_epoch: int = field(default=-1)
    bad_epochs: int = field(default=0)
    observations: int = field(default=0)

    def __post_init__(self) -> None:
        if not isinstance(self.patience, int) or self.patience < 1:
            raise ValueError(f"patience must be an integer >= 1, got {self.patience!r}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta!r}")

    @property
    def has_best(self) -> bool:
        return self.best_epoch >= 0

    @property
    def should_restore(self) -> bool:
        """True when a best-epoch snapshot exists and must be restored."""
        return self.has_best

    def observe(self, value: float, epoch: int) -> bool:
        """Record one validation value; return True when training must stop.

        A non-finite value is rejected loudly — a NaN/Inf validation
        metric means the run is broken and patience must not be burned
        silently.
        """
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(
                f"early stopping requires a finite validation metric, got {value!r} "
                f"at epoch {epoch}"
            )
        self.observations += 1
        if value > self.best_value + self.min_delta:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
