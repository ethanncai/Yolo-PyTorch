"""YOLO 检测数据集（自 Ultralytics data/dataset 精简）。"""

from __future__ import annotations

from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .augment import Compose, Format, LetterBox, v8_transforms
from .base import BaseDataset
from .hyp import TrainHyp
from .instance import Instances
from .utils import (
    DATASET_CACHE_VERSION,
    NUM_THREADS,
    get_hash,
    img2label_paths,
    load_dataset_cache,
    save_dataset_cache,
    verify_image_label,
)


class YOLODataset(BaseDataset):
    def __init__(self, *args, data: dict | None = None, **kwargs):
        self.data = data or {"names": [], "nc": 0}
        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)

    def cache_labels(self, path: Path) -> dict:
        x = {"labels": []}
        nm = nf = ne = nc = 0
        msgs: list[str] = []
        num_cls = len(self.data["names"])
        n = len(self.im_files)
        desc = f"{self.prefix.rstrip()}: scan"
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                verify_image_label,
                zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(num_cls),
                    repeat(self.single_cls),
                ),
            )
            for im_file, lb, shape, _segments, nm_f, nf_f, ne_f, nc_f, msg in tqdm(
                results, total=n, desc=desc, unit="img"
            ):
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],
                            "bboxes": lb[:, 1:5],
                            "person_id": lb[:, 5:6] if lb.shape[1] > 5 else np.full((len(lb), 1), -1, dtype=np.float32),
                            "segments": [],
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
        if nf == 0:
            print(f"{self.prefix}WARNING: no labels found near {path}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = (nf, nm, ne, nc, len(self.im_files))
        x["msgs"] = msgs
        if x["labels"]:
            print(f"{self.prefix}Saving cache -> {path}")
            save_dataset_cache(path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self) -> list[dict]:
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache = load_dataset_cache(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
            print(f"{self.prefix}Loaded cache {cache_path}")
        except (FileNotFoundError, AssertionError, KeyError):
            print(f"{self.prefix}Scanning {len(self.im_files)} labels (cache miss: {cache_path})")
            cache = self.cache_labels(cache_path)

        nf, nm, ne, nc, n = cache.pop("results")
        print(f"{self.prefix}{nf}/{n} images, {nm + ne} backgrounds, {nc} corrupt")
        for msg in cache.get("msgs", []):
            if msg:
                print(msg)
        labels = cache["labels"]
        if not labels:
            raise RuntimeError(f"No valid images in {cache_path}")
        self.im_files = [lb["im_file"] for lb in labels]
        return labels

    def build_transforms(self, hyp: TrainHyp) -> Compose:
        if self.augment:
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        bboxes = label.pop("bboxes")
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        label.pop("segments", None)
        label["instances"] = Instances(bboxes, bbox_format=bbox_format, normalized=normalized)
        label["cls"] = label.pop("cls")
        label["person_id"] = label.pop("person_id", np.full((len(bboxes), 1), -1, dtype=np.float32))
        return label

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        new_batch: dict[str, Any] = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            elif k in {"bboxes", "cls", "person_id"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        if "batch_idx" in new_batch:
            idx_list = list(new_batch["batch_idx"])
            for j in range(len(idx_list)):
                idx_list[j] = idx_list[j] + j
            new_batch["batch_idx"] = torch.cat(idx_list, 0)
        return new_batch
