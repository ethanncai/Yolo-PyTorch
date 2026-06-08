#!/usr/bin/env python3
"""YOLO11 单图推理与可视化：主要逻辑在此文件；依赖可安装的 ``model`` 包。

用法::

    # 项目根目录（或已 ``pip install -e .``）
    python infer.py [ckpt] image.jpg -o out.jpg
    # 省略 ckpt 时默认 ``assets/ckpts/yolo11n.ckpt``

安装后也可用控制台命令：``yolo11-infer``（与本文件 ``main`` 相同）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import YOLO11L, YOLO11M, YOLO11N, YOLO11S, YOLO11X
from model.paths import ckpt_assets_dir
from model.utils import guess_scale_from_name
from model.weights import load_checkpoint_file, load_yolo11_checkpoint

_SCALE_CLS = {"n": YOLO11N, "s": YOLO11S, "m": YOLO11M, "l": YOLO11L, "x": YOLO11X}

# COCO 80 类（与官方 yolo11 预训练一致）
COCO_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _read_ckpt_meta(path: Path) -> tuple[str, int, tuple[str, ...] | None]:
    ck = load_checkpoint_file(path, device="cpu")
    if not isinstance(ck, dict):
        return "", 80, None
    meta = ck.get("meta") or {}
    scale = str(meta.get("scale", "") or "").lower()
    if not scale:
        scale = guess_scale_from_name(path)
    nc = int(meta.get("nc", 80))
    names = meta.get("names")
    names = tuple(str(n) for n in names) if names else None
    return scale, nc, names


def _build_model(ckpt_path: Path, scale: str, nc: int) -> torch.nn.Module:
    if scale not in _SCALE_CLS:
        raise SystemExit(
            f"无法从 checkpoint 解析 scale={scale!r}，请使用 yolo11n/s/m/… 命名的权重或在 meta 中写明 scale。"
        )
    m = _SCALE_CLS[scale](nc=nc)
    load_yolo11_checkpoint(ckpt_path, m, device="cpu", strict=True)
    return m


def letterbox(
    x: torch.Tensor,
    *,
    new_s: int,
    color: float = 114.0 / 255.0,
) -> tuple[torch.Tensor, tuple[float, float], tuple[float, float]]:
    """x: CHW float 0..1。返回 (CHW 正方形 tensor, (ratio, ratio), (pad_left, pad_top))。"""
    _, h, w = x.shape
    r = min(new_s / h, new_s / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        x = F.interpolate(x.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)
    pad_w, pad_h = new_s - nw, new_s - nh
    pl, pr = pad_w // 2, pad_w - pad_w // 2
    pt, pb = pad_h // 2, pad_h - pad_h // 2
    x = F.pad(x, (pl, pr, pt, pb), value=color)
    return x, (r, r), (float(pl), float(pt))


def xywh2xyxy(t: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = t.unbind(-1)
    half_w, half_h = w / 2, h / 2
    return torch.stack((cx - half_w, cy - half_h, cx + half_w, cy + half_h), -1)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N,4) xyxy, b: (M,4) -> (N,M)"""
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp(min=0).prod(-1)
    a_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    b_a = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    union = a_a[:, None] + b_a[None, :] - inter + 1e-7
    return inter / union


def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = box_iou(boxes[i : i + 1], boxes[rest])[0]
        order = rest[iou <= iou_thres]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    iou_thres: float,
    max_det: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_b, out_s, out_c = [], [], []
    for c in cls_ids.unique():
        m = cls_ids == c
        bb, ss, cc = boxes[m], scores[m], cls_ids[m]
        k = nms_xyxy(bb, ss, iou_thres)
        out_b.append(bb[k])
        out_s.append(ss[k])
        out_c.append(cc[k])
    if not out_b:
        d = boxes.device
        return (
            torch.zeros(0, 4, device=d, dtype=boxes.dtype),
            torch.zeros(0, device=d, dtype=scores.dtype),
            torch.zeros(0, device=d, dtype=cls_ids.dtype),
        )
    b = torch.cat(out_b, 0)
    s = torch.cat(out_s, 0)
    c = torch.cat(out_c, 0)
    if b.shape[0] > max_det:
        idx = s.argsort(descending=True)[:max_det]
        b, s, c = b[idx], s[idx], c[idx]
    return b, s, c


def scale_boxes_to_original(
    xyxy: torch.Tensor,
    *,
    ratio: tuple[float, float],
    pad_xy: tuple[float, float],
    orig_w: int,
    orig_h: int,
) -> torch.Tensor:
    padx, pady = pad_xy
    r = ratio[0]
    t = xyxy.clone()
    t[:, [0, 2]] -= padx
    t[:, [1, 3]] -= pady
    t /= r
    t[:, 0::2].clamp_(0, orig_w)
    t[:, 1::2].clamp_(0, orig_h)
    return t


def draw_detections(
    img_rgb: torch.Tensor,
    xyxy: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    names: tuple[str, ...],
) -> "Image.Image":
    from PIL import Image, ImageDraw

    hw = (img_rgb.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
    im = Image.fromarray(hw, mode="RGB")
    dr = ImageDraw.Draw(im)
    ncls = len(names)
    for (x1, y1, x2, y2), sc, ci in zip(xyxy.tolist(), scores.tolist(), cls_ids.tolist()):
        ci = int(ci)
        if not (0 <= ci < ncls):
            label = f"class_{ci}"
        else:
            label = f"{names[ci]} {sc:.2f}"
        dr.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=2)
        dr.text((x1, max(0, y1 - 12)), label, fill=(255, 64, 64))
    return im


def run_infer(
    model: torch.nn.Module,
    img_path: Path,
    *,
    device: torch.device,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    max_nms_candidates: int,
    names: tuple[str, ...] | None = None,
) -> tuple["Image.Image", int]:
    from PIL import Image

    pil = Image.open(img_path).convert("RGB")
    ow, oh = pil.size
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    x_lb, ratio, pad = letterbox(x, new_s=imgsz)
    x_in = x_lb.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        y, _ = model(x_in)
    pred = y[0].transpose(0, 1)
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    conf, cls_id = cls_scores.max(dim=1)
    mask = conf >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]

    if boxes_xywh.numel() == 0:
        return pil.copy(), 0

    if conf.numel() > max_nms_candidates:
        topk = conf.topk(max_nms_candidates).indices
        boxes_xywh = boxes_xywh[topk]
        conf = conf[topk]
        cls_id = cls_id[topk]

    boxes_xyxy = xywh2xyxy(boxes_xywh)
    boxes_xyxy = scale_boxes_to_original(
        boxes_xyxy,
        ratio=ratio,
        pad_xy=pad,
        orig_w=ow,
        orig_h=oh,
    )

    bx, sc, cl = multiclass_nms(boxes_xyxy, conf, cls_id, iou_thres, max_det)
    nc = int(model.nc) if hasattr(model, "nc") else cls_scores.shape[-1]
    if names is None:
        names = COCO_NAMES[:nc] if nc <= len(COCO_NAMES) else tuple(f"c{i}" for i in range(nc))
    img_full = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    vis = draw_detections(img_full, bx.cpu(), sc.cpu(), cl.cpu(), names)
    return vis, int(bx.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO11 .ckpt 单图推理 + 可视化")
    ap.add_argument(
        "ckpt",
        type=Path,
        nargs="?",
        default=None,
        help="yolo11*.ckpt；省略则使用 assets/ckpts/yolo11n.ckpt",
    )
    ap.add_argument("image", type=Path, help="输入图片")
    ap.add_argument("-o", "--output", type=Path, default=None, help="保存路径（默认：<image_stem>_det.jpg）")
    ap.add_argument("--imgsz", type=int, default=640, help="letterbox 边长")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动")
    ap.add_argument("--max-det", type=int, default=300, help="每张图最多保留框数")
    args = ap.parse_args()

    ckpt_path = args.ckpt.expanduser().resolve() if args.ckpt is not None else (ckpt_assets_dir() / "yolo11n.ckpt")
    img_path: Path = args.image.expanduser().resolve()
    ckpt_assets_dir().mkdir(parents=True, exist_ok=True)
    if not ckpt_path.is_file():
        raise SystemExit(f"找不到 ckpt: {ckpt_path}")
    if not img_path.is_file():
        raise SystemExit(f"找不到图片: {img_path}")

    scale, nc, names = _read_ckpt_meta(ckpt_path)
    model = _build_model(ckpt_path, scale, nc)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)

    out_path = args.output
    if out_path is None:
        out_path = img_path.with_name(f"{img_path.stem}_det{img_path.suffix or '.jpg'}")
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis, n = run_infer(
        model,
        img_path,
        device=dev,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        max_nms_candidates=3000,
        names=names,
    )
    vis.save(out_path)
    print(f"saved {out_path} ({n} detections)")


if __name__ == "__main__":
    main()
