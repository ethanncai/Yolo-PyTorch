"""检测验证：在 val 集上跑前向 + NMS + per-class AP，返回 mAP50 / mAP50-95。

与 Ultralytics DetMetrics 对齐：
- NMS：per-class，conf=0.001 / iou=0.7 / max_det=300（默认）
- AP：每个 IoU 阈值（0.50:0.05:0.95，共 10 档）做 101 点插值积分
- fitness = 0.1 * mAP50 + 0.9 * mAP50-95
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

try:
    from torchvision.ops import nms as torchvision_nms
except Exception:  # pragma: no cover - optional speed path
    torchvision_nms = None

__all__ = ["validate", "DetValMetrics"]


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M) IoU。"""
    area1 = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area2 = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    tl = torch.maximum(a[:, None, :2], b[None, :, :2])
    br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp(0).prod(2)
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    if torchvision_nms is not None:
        return torchvision_nms(boxes, scores, iou_thres)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = _box_iou(boxes[i : i + 1], boxes[rest])[0]
        order = rest[iou <= iou_thres]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def _non_max_suppression(
    pred: torch.Tensor,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    max_nms: int = 30000,
) -> torch.Tensor:
    """pred: (anchors, 4+nc)，boxes 为 xywh 像素坐标，scores 已 sigmoid。

    返回 (n,6)：xyxy, conf, cls。
    """
    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:]
    conf, cls = scores.max(1)
    mask = conf >= conf_thres
    if mask.sum() == 0:
        return pred.new_zeros((0, 6))
    boxes = _xywh2xyxy(boxes_xywh[mask])
    conf = conf[mask]
    cls = cls[mask]
    if conf.numel() > max_nms:
        topk = conf.topk(max_nms).indices
        boxes, conf, cls = boxes[topk], conf[topk], cls[topk]
    # per-class NMS：用类偏移避免跨类抑制（与 Ultralytics max_wh 思路一致）
    offset = cls.float() * 7680.0
    keep = _nms(boxes + offset[:, None], conf, iou_thres)
    keep = keep[:max_det]
    return torch.cat([boxes[keep], conf[keep, None], cls[keep, None].float()], 1)


def _match_predictions(
    detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor
) -> torch.Tensor:
    """detections: (N,6) xyxy,conf,cls; labels: (M,5) cls,xyxy。

    返回 (N, len(iouv)) bool TP 矩阵（Ultralytics 匹配规则）。
    """
    n = detections.shape[0]
    correct = torch.zeros((n, iouv.numel()), dtype=torch.bool, device=detections.device)
    if labels.shape[0] == 0 or n == 0:
        return correct
    iou = _box_iou(labels[:, 1:], detections[:, :4])  # (M, N)
    correct_class = labels[:, 0:1] == detections[:, 5]  # (M, N)
    iou = iou * correct_class  # 仅同类匹配
    iou = iou.cpu().numpy()
    for i, thr in enumerate(iouv.cpu().tolist()):
        matches = np.nonzero(iou >= thr)  # (label_idx[], det_idx[])
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                mv = iou[matches[:, 0], matches[:, 1]]
                matches = matches[mv.argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1], i] = True
    return correct


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101 点插值积分（COCO/Ultralytics 风格）。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def _ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    nc: int,
    eps: float = 1e-16,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ap (nc, n_iou) 与 unique_classes 出现标记。"""
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    n_iou = tp.shape[1]
    ap = np.zeros((nc, n_iou), dtype=np.float64)
    seen = np.zeros(nc, dtype=bool)
    for ci, c in enumerate(unique_classes):
        c = int(c)
        i = pred_cls == c
        n_gt = nt[ci]
        if n_gt == 0 or i.sum() == 0:
            continue
        seen[c] = True
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_gt + eps)
        precision = tpc / (tpc + fpc + eps)
        for j in range(n_iou):
            ap[c, j] = _compute_ap(recall[:, j], precision[:, j])
    return ap, seen


_VIZ_PALETTE = (
    (255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 159, 56),
    (56, 255, 255), (255, 56, 255), (255, 255, 56), (128, 128, 128),
)


def _save_val_viz(img: torch.Tensor, det: torch.Tensor, names, out_path: Path) -> None:
    """img: (3,H,W) float 0..1（letterbox 空间）；det: (n,6) xyxy,conf,cls。"""
    from PIL import Image, ImageDraw

    arr = (img.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
    im = Image.fromarray(arr, mode="RGB")
    dr = ImageDraw.Draw(im)
    for x1, y1, x2, y2, sc, ci in det.cpu().tolist():
        ci = int(ci)
        color = _VIZ_PALETTE[ci % len(_VIZ_PALETTE)]
        name = names[ci] if names and 0 <= ci < len(names) else str(ci)
        dr.rectangle([x1, y1, x2, y2], outline=color, width=2)
        dr.text((x1 + 2, max(0, y1 - 11)), f"{name} {sc:.2f}", fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def _save_val_assoc_viz(
    img: torch.Tensor,
    pred: torch.Tensor,
    embeds: "torch.Tensor | None",
    names,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    out_path: Path,
) -> None:
    """画 association 分组：同一 person 同色，挂不上任何人的部位灰色。

    pred: (anchors, 4+nc) xywh 像素 + sigmoid 分数；embeds: (anchors, dim) 或 None。
    复用 infer.py 的聚类与绘制逻辑（letterbox 空间，框已是该尺度）。
    """
    from infer import assign_person_ids, draw_detections, multiclass_nms, xywh2xyxy

    names_t = tuple(names) if names is not None else tuple(str(i) for i in range(int(pred.shape[1]) - 4))
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    conf, cls_id = cls_scores.max(dim=1)
    mask = conf >= conf_thres
    if mask.sum() == 0:
        _save_val_viz(img, pred.new_zeros((0, 6)), names, out_path)
        return
    boxes_xywh, conf, cls_id = boxes_xywh[mask], conf[mask], cls_id[mask]
    emb = embeds[mask] if embeds is not None else None
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    bx, sc, cl, keep_idx = multiclass_nms(boxes_xyxy, conf, cls_id, iou_thres, max_det)
    kept_emb = emb[keep_idx] if emb is not None and keep_idx.numel() else None
    person_ids = assign_person_ids(bx, sc, cl, kept_emb, names_t, assoc_thres=0.45)
    vis = draw_detections(img, bx.cpu(), sc.cpu(), cl.cpu(), names_t, person_ids=person_ids, box_width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)


class DetValMetrics:
    def __init__(self, map50: float, map: float, ap_per_class: np.ndarray, seen: np.ndarray):
        self.map50 = map50
        self.map = map
        self.ap = ap_per_class
        self.seen = seen

    @property
    def fitness(self) -> float:
        return 0.1 * self.map50 + 0.9 * self.map


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    nc: int,
    *,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
    amp: bool = False,
    progress_interval: int = 50,
    viz_dir: "str | Path | None" = None,
    viz_count: int = 8,
    viz_conf: float = 0.25,
    viz_iou: float = 0.45,
    names: "list[str] | None" = None,
) -> DetValMetrics:
    """在 val 集上评估，返回 mAP50 / mAP50-95。GT 与预测均在 letterbox 640 空间比较。

    若提供 ``viz_dir``，则把前 ``viz_count`` 张验证图（含预测框）保存到该目录。
    可视化用 ``viz_conf`` / ``viz_iou``（默认 0.25 / 0.45，与 infer 一致），与
    mAP 评测用的 ``conf_thres`` / ``iou_thres``（0.001 / 0.7，COCO 标准）解耦，
    保证可视化呈现的框与最终推理结果一致。
    """
    was_training = model.training
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats_tp: list[np.ndarray] = []
    stats_conf: list[np.ndarray] = []
    stats_pcls: list[np.ndarray] = []
    target_cls_all: list[np.ndarray] = []

    viz_dir = Path(viz_dir) if viz_dir is not None else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    viz_saved = 0

    total_batches = len(val_loader)
    t0 = time.time()
    print(f"           val: start {len(val_loader.dataset)} images, {total_batches} batches", flush=True)
    for bi, batch in enumerate(val_loader, 1):
        imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
        bs, _, h, w = imgs.shape
        ctx = torch.amp.autocast(device_type="cuda") if (amp and device.type == "cuda") else nullcontext()
        with ctx:
            out = model(imgs)
        preds = out[0] if isinstance(out, (tuple, list)) else out  # (bs, 4+nc, anchors)
        preds = preds.float()
        raw_preds = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        embeds_bchw = None
        if isinstance(raw_preds, dict):
            assoc = raw_preds.get("one2one") if "one2one" in raw_preds else raw_preds
            if isinstance(assoc, dict) and "embeds" in assoc:
                embeds_bchw = assoc["embeds"].float()  # (bs, assoc_dim, anchors)

        batch_idx = batch["batch_idx"].to(device)
        gt_cls = batch["cls"].to(device).view(-1)
        gt_boxes = batch["bboxes"].to(device)  # normalized xywh in letterbox space
        scale = torch.tensor([w, h, w, h], device=device, dtype=gt_boxes.dtype)

        for si in range(bs):
            pred = preds[si].transpose(0, 1)  # (anchors, 4+nc)
            det = _non_max_suppression(pred, conf_thres, iou_thres, max_det)
            gmask = batch_idx == si
            labels = torch.zeros((int(gmask.sum()), 5), device=device)
            if labels.shape[0]:
                labels[:, 0] = gt_cls[gmask]
                labels[:, 1:] = _xywh2xyxy(gt_boxes[gmask] * scale)
            target_cls_all.append(labels[:, 0].cpu().numpy())
            if det.shape[0] == 0:
                continue
            tp = _match_predictions(det, labels, iouv)
            stats_tp.append(tp.cpu().numpy())
            stats_conf.append(det[:, 4].cpu().numpy())
            stats_pcls.append(det[:, 5].cpu().numpy())

        if viz_dir is not None and viz_saved < viz_count:
            for si in range(bs):
                if viz_saved >= viz_count:
                    break
                pred = preds[si].transpose(0, 1)
                det = _non_max_suppression(pred, viz_conf, viz_iou, max_det)
                _save_val_viz(imgs[si], det, names, viz_dir / f"val_pred_{viz_saved:02d}.jpg")
                emb_si = embeds_bchw[si].transpose(0, 1) if embeds_bchw is not None else None
                _save_val_assoc_viz(
                    imgs[si], pred, emb_si, names, viz_conf, viz_iou, max_det,
                    viz_dir / f"val_assoc_{viz_saved:02d}.jpg",
                )
                viz_saved += 1

        if progress_interval > 0 and (bi % progress_interval == 0 or bi == total_batches):
            elapsed = time.time() - t0
            eta = elapsed / max(bi, 1) * (total_batches - bi)
            print(
                f"           val: batch {bi}/{total_batches} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    if was_training:
        model.train()

    target_cls = np.concatenate(target_cls_all, 0) if target_cls_all else np.zeros(0)
    if not stats_tp:
        return DetValMetrics(0.0, 0.0, np.zeros((nc, 10)), np.zeros(nc, dtype=bool))

    tp = np.concatenate(stats_tp, 0)
    conf = np.concatenate(stats_conf, 0)
    pcls = np.concatenate(stats_pcls, 0)
    ap, seen = _ap_per_class(tp, conf, pcls, target_cls, nc)
    if seen.any():
        map50 = float(ap[seen, 0].mean())
        map5095 = float(ap[seen].mean())
    else:
        map50 = map5095 = 0.0
    return DetValMetrics(map50, map5095, ap, seen)
