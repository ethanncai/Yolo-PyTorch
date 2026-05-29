"""标注实例容器（自 Ultralytics utils/instance 精简，仅检测）。"""

from __future__ import annotations

from collections import abc
from itertools import repeat
from numbers import Number

import numpy as np

from .ops import ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh

_FORMATS = ("xyxy", "xywh", "ltwh")


def _ntuple(n: int):
    def parse(x):
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


_to_4tuple = _ntuple(4)


class Bboxes:
    def __init__(self, bboxes: np.ndarray, format: str = "xyxy") -> None:
        assert format in _FORMATS
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2 and bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format

    def convert(self, format: str) -> None:
        assert format in _FORMATS
        if self.format == format:
            return
        if self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self) -> np.ndarray:
        if self.format == "xyxy":
            return (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])
        return self.bboxes[:, 3] * self.bboxes[:, 2]

    def mul(self, scale: Number | tuple | list) -> None:
        if isinstance(scale, Number):
            scale = _to_4tuple(scale)
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset: Number | tuple | list) -> None:
        if isinstance(offset, Number):
            offset = _to_4tuple(offset)
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self) -> int:
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: list[Bboxes], axis: int = 0) -> Bboxes:
        if not boxes_list:
            return cls(np.empty((0, 4)))
        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis), format=boxes_list[0].format)

    def __getitem__(self, index):
        b = self.bboxes[index]
        if isinstance(index, int):
            return Bboxes(b.reshape(1, -1), format=self.format)
        return Bboxes(b, format=self.format)


class Instances:
    def __init__(
        self,
        bboxes: np.ndarray,
        segments: np.ndarray | None = None,
        keypoints: np.ndarray | None = None,
        bbox_format: str = "xywh",
        normalized: bool = True,
    ) -> None:
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments if segments is not None else np.zeros((0, 0, 2), dtype=np.float32)

    def convert_bbox(self, format: str) -> None:
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self) -> np.ndarray:
        return self._bboxes.areas()

    def scale(self, scale_w: float, scale_h: float, bbox_only: bool = False) -> None:
        self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
        if bbox_only or len(self.segments) == 0:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h

    def denormalize(self, w: int, h: int) -> None:
        if not self.normalized:
            return
        self._bboxes.mul(scale=(w, h, w, h))
        if len(self.segments):
            self.segments[..., 0] *= w
            self.segments[..., 1] *= h
        self.normalized = False

    def normalize(self, w: int, h: int) -> None:
        if self.normalized:
            return
        self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        if len(self.segments):
            self.segments[..., 0] /= w
            self.segments[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw: float, padh: float) -> None:
        assert not self.normalized
        self._bboxes.add(offset=(padw, padh, padw, padh))
        if len(self.segments):
            self.segments[..., 0] += padw
            self.segments[..., 1] += padh

    def __getitem__(self, index) -> Instances:
        return Instances(
            bboxes=self.bboxes[index],
            segments=self.segments[index] if len(self.segments) else self.segments,
            keypoints=self.keypoints[index] if self.keypoints is not None else None,
            bbox_format=self._bboxes.format,
            normalized=self.normalized,
        )

    def flipud(self, h: float) -> None:
        if self._bboxes.format == "xyxy":
            y1, y2 = self.bboxes[:, 1].copy(), self.bboxes[:, 3].copy()
            self.bboxes[:, 1] = h - y2
            self.bboxes[:, 3] = h - y1
        else:
            self.bboxes[:, 1] = h - self.bboxes[:, 1]

    def fliplr(self, w: float) -> None:
        if self._bboxes.format == "xyxy":
            x1, x2 = self.bboxes[:, 0].copy(), self.bboxes[:, 2].copy()
            self.bboxes[:, 0] = w - x2
            self.bboxes[:, 2] = w - x1
        else:
            self.bboxes[:, 0] = w - self.bboxes[:, 0]

    def clip(self, w: int, h: int) -> None:
        ori = self._bboxes.format
        self.convert_bbox(format="xyxy")
        self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
        self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
        if ori != "xyxy":
            self.convert_bbox(format=ori)

    def remove_zero_area_boxes(self) -> np.ndarray:
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
        return good

    def __len__(self) -> int:
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: list[Instances], axis: int = 0) -> Instances:
        if not instances_list:
            return cls(np.empty((0, 4)))
        if len(instances_list) == 1:
            return instances_list[0]
        return cls(
            np.concatenate([ins.bboxes for ins in instances_list], axis=axis),
            bbox_format=instances_list[0]._bboxes.format,
            normalized=instances_list[0].normalized,
        )

    @property
    def bboxes(self) -> np.ndarray:
        return self._bboxes.bboxes
