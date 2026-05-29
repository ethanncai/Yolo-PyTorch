"""DataLoader 构建（自 Ultralytics data/build 精简）。"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import YOLODataset
from .hyp import TrainHyp
from .utils import load_data_yaml


class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        while True:
            yield from iter(self.sampler)


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self) -> None:
        self.iterator = self._get_iterator()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset(
    img_path: str,
    data: dict[str, Any],
    *,
    imgsz: int = 640,
    batch_size: int = 16,
    augment: bool = True,
    hyp: TrainHyp | None = None,
    prefix: str = "",
    fraction: float = 1.0,
) -> YOLODataset:
    hyp = hyp or TrainHyp(imgsz=imgsz)
    hyp.imgsz = imgsz
    return YOLODataset(
        img_path=img_path,
        imgsz=imgsz,
        batch_size=batch_size,
        augment=augment,
        hyp=hyp,
        prefix=prefix,
        data=data,
        fraction=fraction,
    )


def build_dataloader(
    dataset,
    batch: int,
    workers: int,
    shuffle: bool = True,
    infinite: bool = True,
) -> DataLoader | InfiniteDataLoader:
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), workers)
    loader_cls = InfiniteDataLoader if infinite else DataLoader
    return loader_cls(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=nw,
        collate_fn=getattr(dataset, "collate_fn", None),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        drop_last=False,
        prefetch_factor=4 if nw > 0 else None,
    )


def build_train_val_loaders(
    data_yaml: str | dict,
    *,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    hyp: TrainHyp | None = None,
) -> tuple[InfiniteDataLoader, DataLoader, dict]:
    data = load_data_yaml(data_yaml) if isinstance(data_yaml, (str, os.PathLike)) else data_yaml
    hyp = hyp or TrainHyp(imgsz=imgsz)
    train_ds = build_yolo_dataset(
        data["train"], data, imgsz=imgsz, batch_size=batch, augment=True, hyp=hyp, prefix="train: "
    )
    val_ds = build_yolo_dataset(
        data["val"], data, imgsz=imgsz, batch_size=batch, augment=False, hyp=hyp, prefix="val: "
    )
    train_loader = build_dataloader(train_ds, batch, workers, shuffle=True, infinite=True)
    val_loader = build_dataloader(val_ds, batch, workers, shuffle=False, infinite=False)
    return train_loader, val_loader, data
