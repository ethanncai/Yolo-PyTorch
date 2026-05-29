"""几何与 bbox 工具（自 Ultralytics data/augment 与 utils/ops 精简移植）。"""

from __future__ import annotations

import numpy as np


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def xywh2ltwh(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2:] = x[..., 2:]
    return y


def ltwh2xywh(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2
    y[..., 1] = x[..., 1] + x[..., 3] / 2
    y[..., 2:] = x[..., 2:]
    return y


def xyxy2ltwh(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def ltwh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y


def segments2boxes(segments: list[np.ndarray]) -> np.ndarray:
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Intersection over area of box2. box1: (N,4) xyxy, box2: (M,4) xyxy -> (N,M)."""
    b1_x1, b1_y1 = box1[:, 0:1], box1[:, 1:2]
    b1_x2, b1_y2 = box1[:, 2:3], box1[:, 3:4]
    b2_x1, b2_y1 = box2[:, 0], box2[:, 1]
    b2_x2, b2_y2 = box2[:, 2], box2[:, 3]

    inter = (
        (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0)
        * (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    )
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
    return inter / area2
