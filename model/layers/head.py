"""YOLO Detect head (v8/v11-style, DFL + decoupled cls/box branches)."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geom import dist2bbox, make_anchors
from .block import DFL
from .conv import Conv, DWConv

__all__ = ["Detect"]


class Detect(nn.Module):
    dynamic = False
    export = False
    format = None
    max_det = 300
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False
    xyxy = False

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = (), assoc_dim: int = 16):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.assoc_dim = assoc_dim
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        c4 = max(16, ch[0] // 4, self.assoc_dim)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.assoc_dim, 1)) for x in ch
        )
        pair_in = self.assoc_dim * 4 + 10
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_in, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3, assoc_head=self.cv4)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, assoc_head=self.one2one_cv4)

    @property
    def end2end(self):
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        self._end2end = value

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
        assoc_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None or assoc_head is None:
            return {}
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        embeds = torch.cat([assoc_head[i](x[i]).view(bs, self.assoc_dim, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, embeds=embeds, feats=x)

    def forward(self, x: list[torch.Tensor]):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int):
        batch_size, anchors, nc = scores.shape
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size, device=scores.device)[..., None], index // nc]
        return scores[..., None], (index % nc)[..., None].float(), idx

    def pair_logits(
        self,
        face_embeds: torch.Tensor,
        person_embeds: torch.Tensor,
        face_boxes: torch.Tensor,
        person_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Score every face-person pair using embeddings plus relative geometry.

        Boxes are xyxy in a shared coordinate system. The geometry terms are scale
        invariant, so train-time letterbox pixels and inference-time original
        pixels both work.
        """
        nf, nperson = face_embeds.shape[0], person_embeds.shape[0]
        if nf == 0 or nperson == 0:
            return face_embeds.new_zeros((nf, nperson))
        f = F.normalize(face_embeds.float(), dim=-1)
        p = F.normalize(person_embeds.float(), dim=-1)
        f_pair = f[:, None, :].expand(nf, nperson, -1)
        p_pair = p[None, :, :].expand(nf, nperson, -1)
        geom = self._pair_geometry(face_boxes.float(), person_boxes.float())
        pair_feat = torch.cat((f_pair, p_pair, (f_pair - p_pair).abs(), f_pair * p_pair, geom), dim=-1)
        pair_feat = pair_feat.to(dtype=next(self.pair_scorer.parameters()).dtype)
        return self.pair_scorer(pair_feat).squeeze(-1)

    @staticmethod
    def _pair_geometry(face_boxes: torch.Tensor, person_boxes: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        fb = face_boxes[:, None, :]
        pb = person_boxes[None, :, :]
        fc = (fb[..., :2] + fb[..., 2:]) / 2
        pc = (pb[..., :2] + pb[..., 2:]) / 2
        fwh = (fb[..., 2:] - fb[..., :2]).clamp(min=eps)
        pwh = (pb[..., 2:] - pb[..., :2]).clamp(min=eps)
        rel_center = (fc - pc) / pwh
        log_wh = torch.log(fwh / pwh)
        inside = torch.stack(
            (
                (fc[..., 0] >= pb[..., 0]).float(),
                (fc[..., 1] >= pb[..., 1]).float(),
                (fc[..., 0] <= pb[..., 2]).float(),
                (fc[..., 1] <= pb[..., 3]).float(),
            ),
            dim=-1,
        )
        lt = torch.maximum(fb[..., :2], pb[..., :2])
        rb = torch.minimum(fb[..., 2:], pb[..., 2:])
        inter = (rb - lt).clamp(min=0).prod(-1, keepdim=True)
        farea = fwh.prod(-1, keepdim=True)
        parea = pwh.prod(-1, keepdim=True)
        io_person = inter / (farea + eps)
        area_ratio = farea / (parea + eps)
        return torch.cat((rel_center, log_wh, inside, io_person, area_ratio), dim=-1)

    def fuse(self) -> None:
        self.cv2 = self.cv3 = self.cv4 = None
