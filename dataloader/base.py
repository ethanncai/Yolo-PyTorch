"""Dataset 基类（自 Ultralytics data/base 精简）。"""

from __future__ import annotations

import glob
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from .hyp import TrainHyp
from .utils import IMG_FORMATS


class BaseDataset(Dataset):
    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        augment: bool = True,
        hyp: TrainHyp | None = None,
        prefix: str = "",
        batch_size: int = 16,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.hyp = hyp or TrainHyp(imgsz=imgsz)
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)
        self.ni = len(self.labels)
        self.batch_size = batch_size
        self.cache = None

        self.buffer: list[int] = []
        self.max_buffer_length = min(self.ni, self.batch_size * 8, 1000) if self.augment else 0
        self.ims: list = [None] * self.ni
        self.im_hw0: list = [None] * self.ni
        self.im_hw: list = [None] * self.ni
        self.transforms = self.build_transforms(self.hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        f: list[str] = []
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p = Path(p)
            if p.is_dir():
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
            elif p.is_file() and p.suffix.lower() == ".txt":
                with open(p, encoding="utf-8") as t:
                    parent = str(p.parent) + os.sep
                    f += [
                        (parent + x.lstrip("./")) if x.startswith("./") else x
                        for x in t.read().strip().splitlines()
                    ]
            elif p.is_file():
                f.append(str(p))
            else:
                raise FileNotFoundError(f"{self.prefix}{p} does not exist")
        im_files = sorted(
            x.replace("/", os.sep)
            for x in f
            if x.rpartition(".")[-1].lower() in IMG_FORMATS
        )
        assert im_files, f"{self.prefix}No images found in {img_path}"
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        if include_class is None:
            if self.single_cls:
                for lb in self.labels:
                    lb["cls"][:, 0] = 0
            return
        inc = np.array(include_class).reshape(1, -1)
        for lb in self.labels:
            j = (lb["cls"] == inc).any(1)
            lb["cls"] = lb["cls"][j]
            lb["bboxes"] = lb["bboxes"][j]
            if self.single_cls:
                lb["cls"][:, 0] = 0

    def load_image(self, i: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        im, f = self.ims[i], self.im_files[i]
        if im is None:
            im = cv2.imread(f, self.cv2_flag)
            if im is None:
                raise FileNotFoundError(f"Image not found: {f}")
            h0, w0 = im.shape[:2]
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if len(self.buffer) > self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = deepcopy(self.labels[index])
        label.pop("shape", None)
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )
        return self.update_labels_info(label)

    def __len__(self) -> int:
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        return label

    def build_transforms(self, hyp: TrainHyp):
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def close_mosaic(self, hyp: TrainHyp) -> None:
        hyp.mosaic = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)
