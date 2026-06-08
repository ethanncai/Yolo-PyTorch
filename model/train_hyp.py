"""Training hyperparameters (loss gains + optimizer schedule)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelHyp:
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    cos_lr: bool = False
    nbs: int = 64
