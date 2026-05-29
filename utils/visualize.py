"""训练 batch 可视化。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# 与 Ultralytics 类似的调色板
_PALETTE = (
    (255, 56, 56), (255, 159, 56), (255, 255, 56), (56, 255, 56), (56, 255, 255),
    (56, 56, 255), (255, 56, 255), (128, 128, 128), (255, 128, 0), (0, 128, 255),
)


def _color(cls_id: int) -> tuple[int, int, int]:
    return _PALETTE[int(cls_id) % len(_PALETTE)]


def _draw_sample(
    img: torch.Tensor,
    bboxes: torch.Tensor,
    cls: torch.Tensor,
    names: list[str],
    *,
    max_boxes: int = 100,
) -> Image.Image:
    """img: (3,H,W) uint8/float; bboxes: (N,4) normalized xywh."""
    arr = img.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)[:, :, ::-1]  # CHW BGR-ish -> RGB for PIL
    h, w = arr.shape[:2]
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    n = min(len(bboxes), max_boxes)
    for i in range(n):
        cx, cy, bw, bh = bboxes[i].tolist()
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        cid = int(cls[i].item()) if cls.ndim else int(cls[i])
        color = _color(cid)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = names[cid] if cid < len(names) else str(cid)
        draw.text((x1, max(0, y1 - 14)), label, fill=color, font=font)
    return pil


def plot_training_batch(
    batch: dict,
    names: list[str],
    save_path: str | Path,
    *,
    max_images: int = 4,
) -> Path:
    """将 collate 后的 batch 画成网格图并保存。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    imgs = batch["img"]
    bboxes = batch["bboxes"]
    cls = batch["cls"]
    batch_idx = batch.get("batch_idx")
    if batch_idx is None:
        raise ValueError("batch 缺少 batch_idx，请使用 YOLODataset.collate_fn")

    tiles: list[Image.Image] = []
    b = min(imgs.shape[0], max_images)
    for i in range(b):
        mask = batch_idx == i
        tiles.append(
            _draw_sample(imgs[i], bboxes[mask], cls[mask], names)
        )

    if not tiles:
        return save_path

    tw, th = tiles[0].size
    cols = min(2, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (tw * cols, th * rows), (114, 114, 114))
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid.paste(tile, (c * tw, r * th))
    grid.save(save_path)
    return save_path
