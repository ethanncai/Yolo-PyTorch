"""数据集扫描、标签校验与 cache（自 Ultralytics data/utils 精简）。"""

from __future__ import annotations

import hashlib
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

from .ops import segments2boxes

IMG_FORMATS = {
    "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp",
    "heic", "heif", "avif", "jp2", "jpeg2000",
}
DATASET_CACHE_VERSION = "1.1.0"
NUM_THREADS = min(8, os.cpu_count() or 1)


def img2label_paths(img_paths: list[str], label_dir: str = "labels") -> list[str]:
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}{label_dir}{os.sep}"
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def get_hash(paths: list[str]) -> str:
    size = sum(os.stat(p).st_size for p in paths if os.path.exists(p))
    h = hashlib.sha256(str(size).encode())
    h.update("".join(paths).encode())
    return h.hexdigest()


def exif_size(img: Image.Image) -> tuple[int, int]:
    s = img.size
    if img.format == "JPEG":
        try:
            if exif := img.getexif():
                if exif.get(274) in {6, 8}:
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def check_image(im_file: str) -> tuple[str, tuple[int, int]]:
    msg = ""
    im = Image.open(im_file)
    im.verify()
    shape = exif_size(im)
    shape = (shape[1], shape[0])
    assert shape[0] > 9 and shape[1] > 9, f"image size {shape} <10 pixels"
    assert im.format and im.format.lower() in IMG_FORMATS, f"invalid format {im.format}"
    if im.format.lower() in {"jpg", "jpeg"}:
        with open(im_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":
                ImageOps.exif_transpose(Image.open(im_file)).save(
                    im_file, "JPEG", subsampling=0, quality=100
                )
                msg = f"{im_file}: corrupt JPEG restored"
    return msg, shape


def verify_image_label(args: tuple) -> tuple:
    im_file, lb_file, prefix, num_cls, single_cls = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, "", []
    try:
        msg, shape = check_image(im_file)
        msg = f"{prefix}{msg}" if msg else ""
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                else:
                    widths = {len(x) for x in lb}
                    assert widths.issubset({5, 6}), f"labels require 5 or 6 columns, got {sorted(widths)}"
                    if len(widths) > 1:
                        lb = [x + ["-1"] if len(x) == 5 else x for x in lb]
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                assert lb.shape[1] in {5, 6}, f"labels require 5 or 6 columns, got {lb.shape[1]}"
                assert lb[:, 1:5].max() <= 1.01, "non-normalized coordinates"
                assert lb[:, :5].min() >= -0.01, "negative labels"
                max_cls = 0 if single_cls else lb[:, 0].max()
                assert max_cls < num_cls, f"class {int(max_cls)} >= nc {num_cls}"
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return None, None, None, None, nm, nf, ne, nc, msg


def load_dataset_cache(path: Path) -> dict:
    import gc

    gc.disable()
    cache = np.load(str(path), allow_pickle=True).item()
    gc.enable()
    return cache


def save_dataset_cache(path: Path, data: dict, version: str) -> None:
    data = dict(data)
    data["version"] = version
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with open(path, "wb") as f:
        np.save(f, data)


def _listify(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _merge_data_yamls(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    if not datasets:
        raise ValueError("no data yaml provided")

    names = list(datasets[0]["names"])
    channels = datasets[0].get("channels", 3)
    for data in datasets[1:]:
        if list(data["names"]) != names:
            raise SyntaxError(
                "multiple data.yaml files must use the same class names and order: "
                f"{datasets[0]['yaml_file']} != {data['yaml_file']}"
            )
        if data.get("channels", 3) != channels:
            raise SyntaxError(
                "multiple data.yaml files must use the same channel count: "
                f"{datasets[0]['yaml_file']} != {data['yaml_file']}"
            )

    merged = dict(datasets[0])
    merged["yaml_file"] = [data["yaml_file"] for data in datasets]
    merged["path"] = [data["path"] for data in datasets]
    merged["names"] = names
    merged["nc"] = len(names)
    merged["channels"] = channels
    for k in ("train", "val", "test"):
        paths: list[str] = []
        for data in datasets:
            if data.get(k):
                paths.extend(_listify(data[k]))
        if paths:
            merged[k] = paths
    return merged


def load_data_yaml(path: str | Path | list[str | Path] | tuple[str | Path, ...]) -> dict[str, Any]:
    """解析 YOLO 数据配置（标准 COCO/YOLO 目录 + data.yaml），支持多个 data.yaml 合并。"""
    if isinstance(path, (list, tuple)):
        return _merge_data_yamls([load_data_yaml(p) for p in path])

    file = Path(path).expanduser().resolve()
    if file.is_dir():
        candidates = list(file.glob("*.yaml")) + list(file.glob("*.yml"))
        if not candidates:
            raise FileNotFoundError(f"no yaml in {file}")
        file = candidates[0]
    if not file.is_file():
        raise FileNotFoundError(file)

    with open(file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["yaml_file"] = str(file)

    for k in ("train", "val"):
        if k not in data:
            if k == "val" and "validation" in data:
                data["val"] = data.pop("validation")
            else:
                raise SyntaxError(f"{file}: missing required key '{k}'")

    if "names" not in data and "nc" not in data:
        raise SyntaxError(f"{file}: need 'names' or 'nc'")
    if "names" in data:
        names = data["names"]
        if isinstance(names, dict):
            data["names"] = [names[i] for i in sorted(names)]
        data["nc"] = len(data["names"])
    else:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]

    root = Path(data.get("path") or file.parent).expanduser()
    if not root.is_absolute():
        root = (file.parent / root).resolve()
    data["path"] = str(root)

    for k in ("train", "val", "test"):
        if not data.get(k):
            continue
        if isinstance(data[k], str):
            data[k] = str((root / data[k]).resolve())
        else:
            data[k] = [str((root / x).resolve()) for x in data[k]]

    data["channels"] = data.get("channels", 3)
    return data
