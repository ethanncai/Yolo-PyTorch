"""训练增强超参（对齐 Ultralytics cfg/default.yaml 检测相关项）。"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class TrainHyp:
    imgsz: int = 640
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    cutmix: float = 0.0
    close_mosaic: int = 10
    bgr: float = 0.0
    mask_ratio: int = 4
    overlap_mask: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> TrainHyp:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
