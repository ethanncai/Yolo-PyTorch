"""Box coordinate ops (Ultralytics-compatible)."""

from __future__ import annotations

import torch

__all__ = ["xywh2xyxy", "xyxy2xywh"]


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2
    y[..., 1] = (y1 + y2) / 2
    y[..., 2] = x2 - x1
    y[..., 3] = y2 - y1
    return y
