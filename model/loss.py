"""YOLOv8/v11 detection loss (from Ultralytics utils/loss.py)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geom import bbox2dist, dist2bbox, make_anchors
from .ops import xywh2xyxy
from .tal import TaskAlignedAssigner

__all__ = ["DFLoss", "BboxLoss", "AssociationLoss", "v8DetectionLoss"]


class DFLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .metrics import bbox_iou

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


class AssociationLoss(nn.Module):
    def __init__(self, face_cls: int = 0, person_cls: int = 1, hand_cls: int = -1) -> None:
        super().__init__()
        self.face_cls = face_cls
        self.person_cls = person_cls
        self.hand_cls = hand_cls

    def forward(
        self,
        pred_embeds: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_person_ids: torch.Tensor,
        target_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        pair_scorer: nn.Module,
    ) -> torch.Tensor:
        loss_terms = []
        batch_size = pred_embeds.shape[0]
        for bi in range(batch_size):
            mask = fg_mask[bi]
            if not mask.any():
                continue
            emb = F.normalize(pred_embeds[bi, mask], dim=-1)
            gt_idx = target_gt_idx[bi, mask]
            obj_embeds, obj_boxes = [], []
            obj_labels: list[int] = []
            obj_pids: list[int] = []
            for gi in gt_idx.unique():
                gi_long = gi.long()
                pid = int(target_person_ids[bi, gi_long])
                if pid < 0:
                    continue
                label = int(target_labels[bi, gi_long])
                part_classes = {self.face_cls}
                if self.hand_cls >= 0:
                    part_classes.add(self.hand_cls)
                if label not in part_classes | {self.person_cls}:
                    continue
                obj_embeds.append(F.normalize(emb[gt_idx == gi].mean(0), dim=0))
                obj_boxes.append(gt_bboxes[bi, gi_long])
                obj_labels.append(label)
                obj_pids.append(pid)

            if len(obj_embeds) < 2:
                continue

            obj = torch.stack(obj_embeds)
            boxes = torch.stack(obj_boxes).to(obj.device)
            labels = torch.tensor(obj_labels, device=obj.device)
            pids = torch.tensor(obj_pids, device=obj.device)
            part_mask = labels == self.face_cls
            if self.hand_cls >= 0:
                part_mask = part_mask | (labels == self.hand_cls)
            person_mask = labels == self.person_cls
            if not part_mask.any() or not person_mask.any():
                continue
            logits = pair_scorer(
                obj[part_mask],
                obj[person_mask],
                boxes[part_mask],
                boxes[person_mask],
            )
            targets = (pids[part_mask, None] == pids[None, person_mask]).to(logits.dtype)
            loss_terms.append(F.binary_cross_entropy_with_logits(logits, targets))
        if not loss_terms:
            return pred_embeds.sum() * 0.0
        return torch.stack(loss_terms).mean()


class v8DetectionLoss:
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.detect_head = m
        self.assoc_loss = AssociationLoss(
            face_cls=int(getattr(h, "face_cls", 0)),
            person_cls=int(getattr(h, "person_cls", 1)),
            hand_cls=int(getattr(h, "hand_cls", -1)),
        ).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)
        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]):
        loss = torch.zeros(4, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        pred_embeds = preds.get("embeds")
        pred_embeds = pred_embeds.permute(0, 2, 1).contiguous() if pred_embeds is not None else None
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        person_id = batch.get("person_id")
        if person_id is None:
            person_id = torch.full_like(batch["cls"].view(-1, 1), -1.0)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"], person_id.view(-1, 1)), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_person_ids = targets.split((1, 4, 1), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            if pred_embeds is not None and (gt_person_ids >= 0).any():
                loss[3] = self.assoc_loss(
                    pred_embeds,
                    fg_mask,
                    target_gt_idx,
                    gt_person_ids.squeeze(-1).long(),
                    gt_labels.squeeze(-1).long(),
                    gt_bboxes,
                    self.detect_head.pair_logits,
                )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= getattr(self.hyp, "assoc", 0.1)
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss.sum() * batch_size, loss_detach

    def __call__(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.loss(preds, batch)
