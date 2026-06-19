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
from model.weights import load_checkpoint_file, load_pretrained_checkpoint, load_yolo11_checkpoint

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


def _read_ckpt_meta(path: Path) -> tuple[str, int, tuple[str, ...] | None, bool]:
    ck = load_checkpoint_file(path, device="cpu")
    if not isinstance(ck, dict):
        return "", 80, None, False
    meta = ck.get("meta") or {}
    scale = str(meta.get("scale", "") or "").lower()
    if not scale:
        scale = guess_scale_from_name(path)
    nc = int(meta.get("nc", 80))
    names = meta.get("names")
    names = tuple(str(n) for n in names) if names else None
    return scale, nc, names, bool(meta)


def _print_ckpt_warning(ckpt_path: Path, nc: int, names: tuple[str, ...] | None, has_meta: bool) -> None:
    if names is None:
        print(
            "warning: checkpoint has no class names in meta; "
            f"visualization will use {'COCO names' if nc <= len(COCO_NAMES) else 'generic class names'}."
        )
    if nc == 80 and names is None and "runs/train" not in ckpt_path.as_posix():
        print(
            "warning: this looks like the default COCO checkpoint, not the 3-class "
            "hand/face/person association model. Use e.g. "
            "runs/train/assoc_test_xhsf_kaggle/weights/best.ckpt for this project."
        )
    if not has_meta:
        print("warning: checkpoint meta is empty; scale/nc were guessed from filename/defaults.")


def _build_model(ckpt_path: Path, scale: str, nc: int) -> torch.nn.Module:
    if scale not in _SCALE_CLS:
        raise SystemExit(
            f"无法从 checkpoint 解析 scale={scale!r}，请使用 yolo11n/s/m/… 命名的权重或在 meta 中写明 scale。"
        )
    m = _SCALE_CLS[scale](nc=nc)
    try:
        load_yolo11_checkpoint(ckpt_path, m, device="cpu", strict=True)
        if hasattr(m.model[-1], "pair_scorer_trained"):
            m.model[-1].pair_scorer_trained = True
    except RuntimeError:
        _, n_loaded, n_total = load_pretrained_checkpoint(ckpt_path, m, device="cpu", reinit_head=False)
        print(f"warning: strict load failed, loaded matching tensors only: {n_loaded}/{n_total}")
        if hasattr(m.model[-1], "pair_scorer"):
            m.model[-1].pair_scorer_trained = False
            print("warning: checkpoint has no compatible pair_scorer; inference will use geometry fallback for association.")
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_b, out_s, out_c, out_i = [], [], [], []
    indices = torch.arange(boxes.shape[0], device=boxes.device)
    for c in cls_ids.unique():
        m = cls_ids == c
        bb, ss, cc = boxes[m], scores[m], cls_ids[m]
        ii = indices[m]
        k = nms_xyxy(bb, ss, iou_thres)
        out_b.append(bb[k])
        out_s.append(ss[k])
        out_c.append(cc[k])
        out_i.append(ii[k])
    if not out_b:
        d = boxes.device
        return (
            torch.zeros(0, 4, device=d, dtype=boxes.dtype),
            torch.zeros(0, device=d, dtype=scores.dtype),
            torch.zeros(0, device=d, dtype=cls_ids.dtype),
            torch.zeros(0, device=d, dtype=torch.long),
        )
    b = torch.cat(out_b, 0)
    s = torch.cat(out_s, 0)
    c = torch.cat(out_c, 0)
    i = torch.cat(out_i, 0)
    if b.shape[0] > max_det:
        idx = s.argsort(descending=True)[:max_det]
        b, s, c, i = b[idx], s[idx], c[idx], i[idx]
    return b, s, c, i


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
    person_ids: list[int] | None = None,
    box_width: int = 5,
) -> "Image.Image":
    from PIL import Image, ImageDraw

    hw = (img_rgb.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
    im = Image.fromarray(hw, mode="RGB")
    dr = ImageDraw.Draw(im)
    ncls = len(names)
    palette = [
        (255, 64, 64),
        (64, 160, 255),
        (80, 220, 120),
        (255, 200, 64),
        (200, 100, 255),
        (64, 220, 220),
        (255, 128, 64),
    ]
    person_ids = person_ids or [-1] * int(xyxy.shape[0])
    for (x1, y1, x2, y2), sc, ci, pid in zip(xyxy.tolist(), scores.tolist(), cls_ids.tolist(), person_ids):
        ci = int(ci)
        color = palette[pid % len(palette)] if pid >= 0 else (128, 128, 128)
        if not (0 <= ci < ncls):
            name = f"class_{ci}"
        else:
            name = names[ci]
        label = f"{name} pid={pid} {sc:.2f}" if pid >= 0 else f"{name} {sc:.2f}"
        dr.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        dr.text((x1 + 2, max(0, y1 - 16)), label, fill=color)
    return im


def _assoc_kind(cls_id: int, names: tuple[str, ...]) -> str:
    name = names[cls_id].lower() if 0 <= cls_id < len(names) else str(cls_id)
    if name in {"person", "body"}:
        return "person"
    if name in {"face", "head"}:
        return "face"
    return "other"


def _kind_exclusive(kind: str) -> bool:
    return kind in {"person", "face"}


def _center(box: torch.Tensor) -> torch.Tensor:
    return (box[:2] + box[2:]) / 2


def _geometry_compatible(box_a: torch.Tensor, kind_a: str, box_b: torch.Tensor, kind_b: str) -> bool:
    if "other" in {kind_a, kind_b}:
        return False
    if kind_a == "person" or kind_b == "person":
        person_box, part_box = (box_a, box_b) if kind_a == "person" else (box_b, box_a)
        x1, y1, x2, y2 = person_box.tolist()
        w, h = x2 - x1, y2 - y1
        px, py = _center(part_box).tolist()
        return (x1 - 0.25 * w) <= px <= (x2 + 0.25 * w) and (y1 - 0.20 * h) <= py <= (y2 + 0.25 * h)
    ca, cb = _center(box_a), _center(box_b)
    diag = torch.linalg.vector_norm(torch.maximum(box_a[2:] - box_a[:2], box_b[2:] - box_b[:2])).item()
    return torch.linalg.vector_norm(ca - cb).item() <= max(diag * 4.0, 80.0)


def assign_person_ids(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    embeds: torch.Tensor | None,
    names: tuple[str, ...],
    assoc_thres: float,
    pair_scorer=None,
) -> list[int]:
    """为每个框分配 person id。

    哲学：person 检测各自作为一个组的种子；face 只有在几何 + embedding
    都与某个 person 兼容时才挂到该组，否则保持 ``-1``（不属于任何人 ->
    推理时画成灰色）。这样 “看起来不属于任何人的部位” 就不管。
    """
    n = int(boxes.shape[0])
    if n == 0:
        return []
    boxes_c = boxes.cpu()
    cls_c = cls_ids.cpu()
    kinds = [_assoc_kind(int(cls_c[i]), names) for i in range(n)]
    emb = F.normalize(embeds.float(), dim=-1).cpu() if embeds is not None and embeds.numel() else None
    pids = [-1] * n
    order = scores.argsort(descending=True).cpu().tolist()

    # 1) 每个 person 检测作为一个组的种子
    person_indices: list[int] = []
    for i in order:
        if kinds[i] == "person":
            pids[i] = len(person_indices)
            person_indices.append(i)

    # 没有任何 person -> 所有部位都不属于任何人，保持灰色
    if not person_indices:
        return pids

    face_indices = [i for i in order if kinds[i] == "face"]
    if not face_indices:
        return pids

    if emb is not None and pair_scorer is not None:
        face_t = torch.tensor(face_indices, dtype=torch.long)
        person_t = torch.tensor(person_indices, dtype=torch.long)
        scorer_owner = getattr(pair_scorer, "__self__", None)
        scorer_device = next(scorer_owner.parameters()).device if scorer_owner is not None else emb.device
        with torch.no_grad():
            pair_probs = pair_scorer(
                emb[face_t].to(scorer_device),
                emb[person_t].to(scorer_device),
                boxes_c[face_t].to(scorer_device),
                boxes_c[person_t].to(scorer_device),
            ).sigmoid().cpu()
        triples = []
        for fi in range(pair_probs.shape[0]):
            for pi in range(pair_probs.shape[1]):
                triples.append((float(pair_probs[fi, pi]), fi, pi))
        used_faces: set[int] = set()
        used_people: set[int] = set()
        for prob, fi, pi in sorted(triples, reverse=True):
            if prob < assoc_thres or fi in used_faces or pi in used_people:
                continue
            pids[face_indices[fi]] = pids[person_indices[pi]]
            used_faces.add(fi)
            used_people.add(pi)
        return pids

    # 2) fallback：没有 pair scorer 时，仅用几何兼容性做一对一分配
    used_people: set[int] = set()
    for i in face_indices:
        best_j, best_sim = -1, -1.0
        for j, person_idx in enumerate(person_indices):
            if j in used_people:
                continue
            if not _geometry_compatible(boxes_c[i], "face", boxes_c[person_idx], "person"):
                continue
            sim = float(scores[i]) + float(scores[person_idx])
            if sim > best_sim:
                best_j, best_sim = j, sim
        if best_j >= 0:
            pids[i] = pids[person_indices[best_j]]
            used_people.add(best_j)
    return pids


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
    assoc_thres: float = 0.45,
    box_width: int = 5,
    keep_names: tuple[str, ...] | None = ("face", "person"),
) -> tuple["Image.Image", int]:
    from PIL import Image

    pil = Image.open(img_path).convert("RGB")
    ow, oh = pil.size
    x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    x_lb, ratio, pad = letterbox(x, new_s=imgsz)
    x_in = x_lb.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        y, raw_preds = model(x_in)
    pred = y[0].transpose(0, 1)
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    raw_embeds = None
    if isinstance(raw_preds, dict):
        assoc_preds = raw_preds.get("one2one") if "one2one" in raw_preds else raw_preds
        if isinstance(assoc_preds, dict) and "embeds" in assoc_preds:
            raw_embeds = assoc_preds["embeds"][0].transpose(0, 1)
    conf, cls_id = cls_scores.max(dim=1)
    mask = conf >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]
    embeds = raw_embeds[mask] if raw_embeds is not None else None

    if boxes_xywh.numel() == 0:
        return pil.copy(), 0

    nc = int(model.nc) if hasattr(model, "nc") else cls_scores.shape[-1]
    if names is None:
        names = COCO_NAMES[:nc] if nc <= len(COCO_NAMES) else tuple(f"c{i}" for i in range(nc))

    if keep_names:
        keep_set = {name.lower() for name in keep_names}
        keep_cls = torch.tensor(
            [i for i, name in enumerate(names) if name.lower() in keep_set],
            device=cls_id.device,
            dtype=cls_id.dtype,
        )
        if keep_cls.numel() == 0:
            print(f"warning: none of --keep-names {tuple(keep_names)} exist in checkpoint names={names}")
            return pil.copy(), 0
        keep_mask = (cls_id[:, None] == keep_cls[None, :]).any(1)
        boxes_xywh = boxes_xywh[keep_mask]
        conf = conf[keep_mask]
        cls_id = cls_id[keep_mask]
        embeds = embeds[keep_mask] if embeds is not None else None

    if boxes_xywh.numel() == 0:
        return pil.copy(), 0

    if conf.numel() > max_nms_candidates:
        topk = conf.topk(max_nms_candidates).indices
        boxes_xywh = boxes_xywh[topk]
        conf = conf[topk]
        cls_id = cls_id[topk]
        embeds = embeds[topk] if embeds is not None else None

    boxes_xyxy = xywh2xyxy(boxes_xywh)
    boxes_xyxy = scale_boxes_to_original(
        boxes_xyxy,
        ratio=ratio,
        pad_xy=pad,
        orig_w=ow,
        orig_h=oh,
    )

    bx, sc, cl, keep_idx = multiclass_nms(boxes_xyxy, conf, cls_id, iou_thres, max_det)
    kept_embeds = embeds[keep_idx] if embeds is not None and keep_idx.numel() else None
    head = model.model[-1] if hasattr(model, "model") else None
    pair_scorer = getattr(head, "pair_logits", None) if getattr(head, "pair_scorer_trained", True) else None
    person_ids = assign_person_ids(bx, sc, cl, kept_embeds, names, assoc_thres, pair_scorer=pair_scorer)
    img_full = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
    vis = draw_detections(img_full, bx.cpu(), sc.cpu(), cl.cpu(), names, person_ids=person_ids, box_width=box_width)
    return vis, int(bx.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO11 .ckpt 单图推理 + 可视化")
    ap.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="输入图片，或 ckpt + 输入图片；只给图片时使用 assets/ckpts/yolo11n.ckpt",
    )
    ap.add_argument("-o", "--output", type=Path, default=None, help="保存路径（默认：<image_stem>_det.jpg）")
    ap.add_argument("--imgsz", type=int, default=640, help="letterbox 边长")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动")
    ap.add_argument("--max-det", type=int, default=300, help="每张图最多保留框数")
    ap.add_argument("--assoc-thres", type=float, default=0.45, help="association embedding 聚类阈值")
    ap.add_argument("--box-width", type=int, default=5, help="可视化框线宽")
    ap.add_argument(
        "--keep-names",
        type=str,
        default="face,person",
        help="推理时保留的类别名，默认 face,person；传空字符串表示保留 checkpoint 全部类别",
    )
    args = ap.parse_args()

    if len(args.paths) == 1:
        ckpt_path = ckpt_assets_dir() / "yolo11n.ckpt"
        img_path = args.paths[0].expanduser().resolve()
    elif len(args.paths) == 2:
        ckpt_path = args.paths[0].expanduser().resolve()
        img_path = args.paths[1].expanduser().resolve()
    else:
        raise SystemExit("用法: python infer.py [ckpt] image.jpg -o out.jpg")
    ckpt_assets_dir().mkdir(parents=True, exist_ok=True)
    if not ckpt_path.is_file():
        raise SystemExit(f"找不到 ckpt: {ckpt_path}")
    if not img_path.is_file():
        raise SystemExit(f"找不到图片: {img_path}")

    scale, nc, names, has_meta = _read_ckpt_meta(ckpt_path)
    _print_ckpt_warning(ckpt_path, nc, names, has_meta)
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
        assoc_thres=args.assoc_thres,
        box_width=args.box_width,
        keep_names=tuple(x.strip() for x in args.keep_names.split(",") if x.strip()) or None,
    )
    vis.save(out_path)
    print(f"saved {out_path} ({n} detections)")


if __name__ == "__main__":
    main()
