"""数据加载与增强（Ultralytics-free，逻辑对齐 YOLOv8/v11）。"""

from .augment import Compose, LetterBox, v8_transforms
from .base import BaseDataset
from .build import InfiniteDataLoader, build_dataloader, build_train_val_loaders, build_yolo_dataset
from .dataset import YOLODataset
from .hyp import TrainHyp
from .utils import load_data_yaml

__all__ = [
    "BaseDataset",
    "YOLODataset",
    "TrainHyp",
    "Compose",
    "LetterBox",
    "v8_transforms",
    "InfiniteDataLoader",
    "build_yolo_dataset",
    "build_dataloader",
    "build_train_val_loaders",
    "load_data_yaml",
]
