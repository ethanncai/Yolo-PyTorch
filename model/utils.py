"""Small helpers (no Ultralytics dependency)."""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["make_divisible", "guess_scale_from_name", "initialize_weights"]


def make_divisible(x: int, divisor: int) -> int:
    """Return x rounded up to nearest multiple of divisor (Ultralytics-style)."""
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


def guess_scale_from_name(path: str | Path) -> str:
    """Parse n/s/m/l/x from filename stem, e.g. yolo11n.pt -> n."""
    try:
        return re.search(r"yolo(?:e-)?[v]?\d+([nslmx])", Path(path).stem, re.I).group(1).lower()
    except (AttributeError, IndexError):
        return ""


def initialize_weights(model) -> None:
    """Match Ultralytics defaults for BN eps/momentum and activation inplace flags."""
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif type(m) in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True
