#!/usr/bin/env python3
"""Video inference with person tracking and passive part tracking.

Runs the YOLO model on small frame batches. Persons are tracked by ByteTrack;
faces/hands inherit the track id and color of the matched person from the part-person
pair scorer.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
try:
    from torchvision.ops import batched_nms as torchvision_batched_nms
except Exception:
    torchvision_batched_nms = None

from infer import (
    COCO_NAMES,
    _build_model,
    _read_ckpt_meta,
    _print_ckpt_warning,
    box_iou,
    multiclass_nms,
    scale_boxes_to_original,
    xywh2xyxy,
)


@dataclass
class ByteTrackArgs:
    track_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer: int = 30


@dataclass
class ByteTrackTrack:
    track_id: int
    tlbr: np.ndarray
    score: float
    embed: torch.Tensor | None = None
    age: int = 0
    hits: int = 1
    state: str = "tracked"


class ByteTracker:
    """Small self-contained ByteTrack implementation for person boxes.

    It keeps ByteTrack's important behavior: first match high-score detections,
    then recover unmatched tracks with low-score detections before declaring
    them lost. This is the part a plain IoU tracker was missing.
    """

    def __init__(self, args: ByteTrackArgs, frame_rate: float = 30.0) -> None:
        self.args = args
        self.max_time_lost = max(1, int(frame_rate / 30.0 * args.track_buffer))
        self.tracked: list[ByteTrackTrack] = []
        self.lost: list[ByteTrackTrack] = []
        self.next_id = 0

    def update(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        embeds: torch.Tensor | None = None,
    ) -> list[ByteTrackTrack]:
        boxes_np = boxes.detach().cpu().numpy().astype(np.float32) if boxes.numel() else np.zeros((0, 4), dtype=np.float32)
        scores_np = scores.detach().cpu().numpy().astype(np.float32) if scores.numel() else np.zeros((0,), dtype=np.float32)
        embeds_cpu = embeds.detach().cpu() if embeds is not None and embeds.numel() else None

        detections = [self._make_det(i, boxes_np[i], float(scores_np[i]), embeds_cpu) for i in range(len(scores_np))]
        high = [d for d in detections if d.score >= self.args.track_thresh]
        low = [d for d in detections if self.args.track_low_thresh <= d.score < self.args.track_thresh]

        track_pool = self.tracked + self.lost
        matches, unmatched_tracks, unmatched_high = self._match(track_pool, high, self.args.match_thresh, fuse_score=True)

        activated: list[ByteTrackTrack] = []
        refind: list[ByteTrackTrack] = []
        for track_idx, det_idx in matches:
            track = track_pool[track_idx]
            self._update_track(track, high[det_idx])
            if track in self.lost:
                refind.append(track)
            else:
                activated.append(track)

        remaining_tracks = [track_pool[i] for i in unmatched_tracks if track_pool[i].state == "tracked"]
        matches2, unmatched_remaining, _ = self._match(remaining_tracks, low, 0.5, fuse_score=False)
        for track_idx, det_idx in matches2:
            track = remaining_tracks[track_idx]
            self._update_track(track, low[det_idx])
            activated.append(track)

        newly_lost = []
        for idx in unmatched_remaining:
            track = remaining_tracks[idx]
            track.state = "lost"
            track.age = 0
            newly_lost.append(track)

        for det_idx in unmatched_high:
            det = high[det_idx]
            if det.score < self.args.new_track_thresh:
                continue
            det.track_id = self.next_id
            self.next_id += 1
            det.state = "tracked"
            activated.append(det)

        old_tracked = [t for t in self.tracked if t.state == "tracked" and t not in activated]
        old_lost = [t for t in self.lost if t.state == "lost" and t not in refind]
        for track in old_lost + newly_lost:
            track.age += 1

        self.tracked = self._dedupe_tracks(old_tracked + activated + refind)
        self.lost = [t for t in self._dedupe_tracks(old_lost + newly_lost) if t.age <= self.max_time_lost]
        return list(self.tracked)

    def _make_det(
        self,
        det_idx: int,
        tlbr: np.ndarray,
        score: float,
        embeds: torch.Tensor | None,
    ) -> ByteTrackTrack:
        embed = embeds[det_idx].clone() if embeds is not None else None
        return ByteTrackTrack(track_id=-1, tlbr=tlbr.copy(), score=score, embed=embed)

    @staticmethod
    def _update_track(track: ByteTrackTrack, det: ByteTrackTrack) -> None:
        track.tlbr = det.tlbr.copy()
        track.score = det.score
        track.embed = det.embed.clone() if det.embed is not None else track.embed
        track.age = 0
        track.hits += 1
        track.state = "tracked"

    def _match(
        self,
        tracks: list[ByteTrackTrack],
        detections: list[ByteTrackTrack],
        thresh: float,
        *,
        fuse_score: bool,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = torch.tensor(np.stack([t.tlbr for t in tracks]), dtype=torch.float32)
        det_boxes = torch.tensor(np.stack([d.tlbr for d in detections]), dtype=torch.float32)
        iou = box_iou(track_boxes, det_boxes).numpy()
        cost = 1.0 - iou
        if fuse_score:
            det_scores = np.asarray([d.score for d in detections], dtype=np.float32)[None, :]
            cost = 1.0 - iou * det_scores
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if cost[r, c] > thresh:
                continue
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)
        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)

    @staticmethod
    def _dedupe_tracks(tracks: list[ByteTrackTrack]) -> list[ByteTrackTrack]:
        by_id: dict[int, ByteTrackTrack] = {}
        for track in tracks:
            prev = by_id.get(track.track_id)
            if prev is None or track.hits >= prev.hits:
                by_id[track.track_id] = track
        return list(by_id.values())


class SourceByteTrackerAdapter:
    def __init__(self, args: ByteTrackArgs, frame_rate: float, frame_shape: tuple[int, int]) -> None:
        byte_root = Path("/home/junzhicai/ByteTrack")
        if str(byte_root) not in sys.path:
            sys.path.insert(0, str(byte_root))
        if not hasattr(np, "float"):
            np.float = float  # type: ignore[attr-defined]
        from yolox.tracker.byte_tracker import BYTETracker as SourceBYTETracker

        self.frame_shape = frame_shape
        source_args = SimpleNamespace(
            track_thresh=args.track_thresh,
            track_buffer=args.track_buffer,
            match_thresh=args.match_thresh,
            mot20=False,
        )
        self.tracker = SourceBYTETracker(source_args, frame_rate=frame_rate)
        self.tracker.det_thresh = args.new_track_thresh

    def update(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        embeds: torch.Tensor | None = None,
    ) -> list[ByteTrackTrack]:
        if boxes.numel() == 0:
            dets = np.zeros((0, 5), dtype=np.float32)
        else:
            dets = torch.cat((boxes.detach().cpu(), scores.detach().cpu().view(-1, 1)), dim=1).numpy().astype(np.float32)
        img_h, img_w = self.frame_shape
        tracks = self.tracker.update(dets, (img_h, img_w), (img_h, img_w))
        return [
            ByteTrackTrack(
                track_id=int(t.track_id),
                tlbr=np.asarray(t.tlbr, dtype=np.float32).copy(),
                score=float(t.score),
                state="tracked",
            )
            for t in tracks
        ]


def build_person_tracker(args: ByteTrackArgs, frame_rate: float, frame_shape: tuple[int, int]):
    try:
        tracker = SourceByteTrackerAdapter(args, frame_rate, frame_shape)
        print("tracker: using /home/junzhicai/ByteTrack source BYTETracker")
        return tracker
    except Exception as exc:
        print(f"warning: source BYTETracker unavailable ({type(exc).__name__}: {exc}); using built-in ByteTrack fallback")
        return ByteTracker(args, frame_rate=frame_rate)


def color_for_track(track_id: int) -> tuple[int, int, int]:
    if track_id < 0:
        return (150, 150, 150)
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_label(frame: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    x0 = max(0, x)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 6, y0 + th + baseline + 4), color, -1)
    cv2.putText(frame, text, (x0 + 3, y0 + th + 1), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def fast_letterbox_frame(
    frame_bgr: np.ndarray,
    imgsz: int,
) -> tuple[torch.Tensor, tuple[float, float], tuple[float, float]]:
    h, w = frame_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame_bgr
    pad_w, pad_h = imgsz - nw, imgsz - nh
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    return x, (r, r), (float(left), float(top))


def fast_multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    iou_thres: float,
    max_det: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        d = boxes.device
        return (
            torch.zeros(0, 4, device=d, dtype=boxes.dtype),
            torch.zeros(0, device=d, dtype=scores.dtype),
            torch.zeros(0, device=d, dtype=cls_ids.dtype),
            torch.zeros(0, device=d, dtype=torch.long),
        )
    if torchvision_batched_nms is None:
        return multiclass_nms(boxes, scores, cls_ids, iou_thres, max_det)
    keep = torchvision_batched_nms(boxes.float(), scores.float(), cls_ids, iou_thres)
    if keep.numel() > max_det:
        keep = keep[:max_det]
    return boxes[keep], scores[keep], cls_ids[keep], keep


def draw_tracks(
    frame: np.ndarray,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    names: tuple[str, ...],
    track_ids: list[int],
    person_mask: torch.Tensor,
    box_width: int,
) -> np.ndarray:
    out = frame.copy()
    for idx, ((x1, y1, x2, y2), score, cls_id, track_id) in enumerate(
        zip(boxes.cpu().tolist(), scores.cpu().tolist(), cls_ids.cpu().tolist(), track_ids)
    ):
        cls_id = int(cls_id)
        name = names[cls_id] if 0 <= cls_id < len(names) else f"class_{cls_id}"
        is_person = bool(person_mask[idx])
        color = color_for_track(track_id)
        x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), (x1, y1, x2, y2))
        width = box_width if is_person else max(1, box_width - 2)
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, width)
        if not is_person:
            cx = int(round((x1 + x2) / 2))
            cy = int(round((y1 + y2) / 2))
            cv2.circle(out, (cx, cy), 3, color, -1)
        tag = f"T{track_id} {name} {score:.2f}" if track_id >= 0 else f"unmatched {name} {score:.2f}"
        draw_label(out, x1i + 2, y1i - 2, tag, color)
    return out


@torch.no_grad()
def detect_frame(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    *,
    device: torch.device,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    max_nms_candidates: int,
    names: tuple[str, ...],
    keep_names: tuple[str, ...] | None,
    min_conf: float | None = None,
    half: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return detect_frames(
        model,
        [frame_bgr],
        device=device,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=max_det,
        max_nms_candidates=max_nms_candidates,
        names=names,
        keep_names=keep_names,
        min_conf=min_conf,
        half=half,
    )[0]


@torch.no_grad()
def detect_frames(
    model: torch.nn.Module,
    frames_bgr: list[np.ndarray],
    *,
    device: torch.device,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    max_nms_candidates: int,
    names: tuple[str, ...],
    keep_names: tuple[str, ...] | None,
    min_conf: float | None = None,
    half: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    if not frames_bgr:
        return []
    original_shapes = [frame.shape[:2] for frame in frames_bgr]
    preprocessed, ratios, pads = [], [], []
    for frame in frames_bgr:
        x_lb, ratio, pad = fast_letterbox_frame(frame, imgsz)
        preprocessed.append(x_lb)
        ratios.append(ratio)
        pads.append(pad)
    batch = torch.stack(preprocessed, 0).to(device, non_blocking=True)
    if half:
        batch = batch.half()
    y, raw_preds = model(batch)
    raw_embeds_all = None
    if isinstance(raw_preds, dict):
        assoc_preds = raw_preds.get("one2one") if "one2one" in raw_preds else raw_preds
        if isinstance(assoc_preds, dict) and "embeds" in assoc_preds:
            raw_embeds_all = assoc_preds["embeds"]

    keep_cls = None
    if keep_names:
        keep_set = {name.lower() for name in keep_names}
        keep_cls = torch.tensor([i for i, name in enumerate(names) if name.lower() in keep_set], device=device)

    results = []
    for batch_idx, (oh, ow) in enumerate(original_shapes):
        pred = y[batch_idx].transpose(0, 1)
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        conf, cls_id = cls_scores.max(dim=1)
        raw_embeds = raw_embeds_all[batch_idx].transpose(0, 1) if raw_embeds_all is not None else None
        mask = conf >= (conf_thres if min_conf is None else min_conf)
        boxes_xywh, conf, cls_id = boxes_xywh[mask], conf[mask], cls_id[mask]
        embeds = raw_embeds[mask] if raw_embeds is not None else None
        if boxes_xywh.numel() == 0:
            results.append((_empty(device), conf.new_zeros(0), cls_id.new_zeros(0), None))
            continue
        if keep_cls is not None:
            keep = (cls_id[:, None] == keep_cls[None, :]).any(1) if keep_cls.numel() else torch.zeros_like(cls_id, dtype=torch.bool)
            boxes_xywh, conf, cls_id = boxes_xywh[keep], conf[keep], cls_id[keep]
            embeds = embeds[keep] if embeds is not None else None
        if boxes_xywh.numel() == 0:
            results.append((_empty(device), conf.new_zeros(0), cls_id.new_zeros(0), None))
            continue
        if conf.numel() > max_nms_candidates:
            topk = conf.topk(max_nms_candidates).indices
            boxes_xywh, conf, cls_id = boxes_xywh[topk], conf[topk], cls_id[topk]
            embeds = embeds[topk] if embeds is not None else None
        boxes_xyxy = scale_boxes_to_original(xywh2xyxy(boxes_xywh), ratio=ratios[batch_idx], pad_xy=pads[batch_idx], orig_w=ow, orig_h=oh)
        boxes, scores, classes, keep_idx = fast_multiclass_nms(boxes_xyxy, conf, cls_id, iou_thres, max_det)
        kept_embeds = embeds[keep_idx] if embeds is not None and keep_idx.numel() else None
        results.append((boxes, scores, classes, kept_embeds))
    return results


def _empty(device: torch.device) -> torch.Tensor:
    return torch.zeros(0, 4, device=device)


def _part_capacity(kind: str) -> int:
    return 2 if kind in {"hand", "left_hand", "right_hand"} else 1


def match_parts_to_tracks(
    model: torch.nn.Module,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    embeds: torch.Tensor | None,
    names: tuple[str, ...],
    person_track_ids: list[int],
    assoc_thres: float,
) -> tuple[list[int], torch.Tensor]:
    kinds = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in cls_ids.cpu().tolist()]
    person_mask = torch.tensor([k in {"person", "body"} for k in kinds], dtype=torch.bool)
    part_mask = torch.tensor([k in {"face", "head", "hand", "left_hand", "right_hand"} for k in kinds], dtype=torch.bool)
    out_ids = [-1] * int(boxes.shape[0])
    person_indices = torch.nonzero(person_mask).flatten().tolist()
    part_indices = torch.nonzero(part_mask).flatten().tolist()
    for det_idx in person_indices:
        out_ids[det_idx] = person_track_ids[det_idx] if det_idx < len(person_track_ids) else -1
    if not part_indices or not person_indices:
        return out_ids, person_mask
    pair_scorer = None
    head = model.model[-1] if hasattr(model, "model") else None
    if getattr(head, "pair_scorer_trained", True):
        pair_scorer = getattr(head, "pair_logits", None)
    if embeds is not None and pair_scorer is not None:
        scorer_owner = getattr(pair_scorer, "__self__", None)
        scorer_param = next(scorer_owner.parameters()) if scorer_owner is not None else None
        scorer_device = scorer_param.device if scorer_param is not None else embeds.device
        scorer_dtype = scorer_param.dtype if scorer_param is not None else embeds.dtype
        part_t = torch.tensor(part_indices, dtype=torch.long)
        person_t = torch.tensor(person_indices, dtype=torch.long)
        pair_probs = pair_scorer(
            embeds[part_t].to(device=scorer_device, dtype=scorer_dtype),
            embeds[person_t].to(device=scorer_device, dtype=scorer_dtype),
            boxes[part_t].to(device=scorer_device, dtype=scorer_dtype),
            boxes[person_t].to(device=scorer_device, dtype=scorer_dtype),
        ).sigmoid().cpu()
        triples = [(float(pair_probs[fi, pi]), fi, pi) for fi in range(pair_probs.shape[0]) for pi in range(pair_probs.shape[1])]
    else:
        ious = box_iou(boxes[part_indices].cpu(), boxes[person_indices].cpu())
        triples = [(float(ious[fi, pi]), fi, pi) for fi in range(ious.shape[0]) for pi in range(ious.shape[1])]
    used_parts: set[int] = set()
    person_counts = [0] * len(person_indices)
    for score, fi, pi in sorted(triples, reverse=True):
        kind = kinds[part_indices[fi]]
        if score < assoc_thres or fi in used_parts or person_counts[pi] >= _part_capacity(kind):
            continue
        person_det_idx = person_indices[pi]
        out_ids[part_indices[fi]] = person_track_ids[person_det_idx] if person_det_idx < len(person_track_ids) else -1
        used_parts.add(fi)
        person_counts[pi] += 1
    return out_ids, person_mask


def inherit_unmatched_parts_from_tracks(
    track_ids: list[int],
    boxes: torch.Tensor,
    cls_ids: torch.Tensor,
    names: tuple[str, ...],
    tracks: list[ByteTrackTrack],
) -> list[int]:
    if not tracks or boxes.numel() == 0:
        return track_ids
    kinds = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in cls_ids.detach().cpu().tolist()]
    part_indices = [i for i, k in enumerate(kinds) if k in {"face", "head", "hand", "left_hand", "right_hand"} and track_ids[i] < 0]
    if not part_indices:
        return track_ids
    track_boxes = torch.tensor(np.stack([t.tlbr for t in tracks]), dtype=torch.float32)
    part_boxes = boxes[part_indices].detach().cpu()
    part_centers = (part_boxes[:, :2] + part_boxes[:, 2:]) / 2
    tb = track_boxes
    tw = (tb[:, 2] - tb[:, 0]).clamp(min=1)
    th = (tb[:, 3] - tb[:, 1]).clamp(min=1)
    expanded = torch.stack((tb[:, 0] - 0.25 * tw, tb[:, 1] - 0.15 * th, tb[:, 2] + 0.25 * tw, tb[:, 3] + 0.25 * th), dim=1)
    track_counts = [0] * len(tracks)
    candidates = []
    for fi, center in enumerate(part_centers):
        inside = (center[0] >= expanded[:, 0]) & (center[1] >= expanded[:, 1]) & (center[0] <= expanded[:, 2]) & (center[1] <= expanded[:, 3])
        for ti in torch.nonzero(inside).flatten().tolist():
            tx1, ty1, tx2, ty2 = tb[ti].tolist()
            kind = kinds[part_indices[fi]]
            y_ratio = 0.18 if kind in {"face", "head"} else 0.55
            tc = torch.tensor([(tx1 + tx2) / 2, ty1 + (ty2 - ty1) * y_ratio])
            dist = float(torch.linalg.vector_norm(center - tc) / max((tw[ti] ** 2 + th[ti] ** 2).sqrt().item(), 1.0))
            candidates.append((dist, fi, ti))
    for _dist, fi, ti in sorted(candidates):
        kind = kinds[part_indices[fi]]
        if track_counts[ti] >= _part_capacity(kind):
            continue
        track_ids[part_indices[fi]] = tracks[ti].track_id
        track_counts[ti] += 1
    return track_ids


def visible_and_track_inputs(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    cls_ids: torch.Tensor,
    embeds: torch.Tensor | None,
    names: tuple[str, ...],
    vis_conf: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    kinds = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in cls_ids.detach().cpu().tolist()]
    person_mask = torch.tensor([k in {"person", "body"} for k in kinds], dtype=torch.bool, device=boxes.device)
    vis_mask = scores >= vis_conf
    vis_boxes, vis_scores, vis_cls = boxes[vis_mask], scores[vis_mask], cls_ids[vis_mask]
    vis_embeds = embeds[vis_mask] if embeds is not None else None
    person_boxes = boxes[person_mask]
    person_scores = scores[person_mask]
    person_embeds = embeds[person_mask] if embeds is not None else None
    return vis_boxes, vis_scores, vis_cls, vis_embeds, person_boxes, person_scores, person_embeds


def assign_person_tracks_to_visible(
    visible_boxes: torch.Tensor,
    visible_cls: torch.Tensor,
    names: tuple[str, ...],
    tracks: list[ByteTrackTrack],
    iou_thres: float,
) -> list[int]:
    kinds = [names[int(c)].lower() if 0 <= int(c) < len(names) else str(int(c)) for c in visible_cls.detach().cpu().tolist()]
    person_indices = [i for i, k in enumerate(kinds) if k in {"person", "body"}]
    ids = [-1] * int(visible_boxes.shape[0])
    if not person_indices or not tracks:
        return ids
    track_boxes = torch.tensor(np.stack([t.tlbr for t in tracks]), dtype=torch.float32)
    ious = box_iou(visible_boxes[person_indices].detach().cpu(), track_boxes)
    used_dets: set[int] = set()
    used_tracks: set[int] = set()
    triples = [(float(ious[di, ti]), di, ti) for di in range(ious.shape[0]) for ti in range(ious.shape[1])]
    for iou, det_pos, track_pos in sorted(triples, reverse=True):
        if iou < iou_thres or det_pos in used_dets or track_pos in used_tracks:
            continue
        ids[person_indices[det_pos]] = tracks[track_pos].track_id
        used_dets.add(det_pos)
        used_tracks.add(track_pos)
    return ids


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "XVID"))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open output video writer: {path}")
    return writer


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO face/person video inference with person tracking")
    ap.add_argument("ckpt", type=Path, help="trained checkpoint")
    ap.add_argument("video", type=Path, help="input video")
    ap.add_argument("-o", "--output", type=Path, default=None, help="output video path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--assoc-thres", type=float, default=0.45)
    ap.add_argument("--track-low-conf", type=float, default=0.1, help="minimum person confidence fed to ByteTrack")
    ap.add_argument("--track-high-conf", type=float, default=0.5, help="ByteTrack high-score association threshold")
    ap.add_argument("--new-track-conf", type=float, default=0.6, help="minimum high-score person confidence to start a new track")
    ap.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack first-stage matching cost threshold")
    ap.add_argument("--track-buffer", type=int, default=30, help="frames to keep lost tracks")
    ap.add_argument("--track-assign-iou", type=float, default=0.3, help="IoU used to map ByteTrack tracks back to visible person boxes")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--half", action="store_true", help="use FP16 inference on CUDA")
    ap.add_argument("--batch-size", type=int, default=4, help="frames per model forward pass")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--max-nms-candidates", type=int, default=1000, help="top scoring candidates kept before NMS per frame")
    ap.add_argument("--box-width", type=int, default=4)
    ap.add_argument("--keep-names", type=str, default="face,person")
    args = ap.parse_args()

    ckpt_path = args.ckpt.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"找不到 ckpt: {ckpt_path}")
    if not video_path.is_file():
        raise SystemExit(f"找不到视频: {video_path}")

    scale, nc, names, has_meta = _read_ckpt_meta(ckpt_path)
    _print_ckpt_warning(ckpt_path, nc, names, has_meta)
    names = names or (COCO_NAMES[:nc] if nc <= len(COCO_NAMES) else tuple(f"c{i}" for i in range(nc)))
    model = _build_model(ckpt_path, scale, nc)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_half = bool(args.half and device.type == "cuda")
    if use_half:
        model = model.half()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = args.output.expanduser().resolve() if args.output else video_path.with_name(f"{video_path.stem}_tracked.mp4")
    writer = open_writer(out_path, fps, width, height)

    tracker_args = ByteTrackArgs(
        track_thresh=args.track_high_conf,
        track_low_thresh=args.track_low_conf,
        new_track_thresh=args.new_track_conf,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
    )
    tracker = build_person_tracker(tracker_args, frame_rate=fps, frame_shape=(height, width))
    keep_names = tuple(x.strip() for x in args.keep_names.split(",") if x.strip()) or None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    t0 = time.perf_counter()
    try:
        while True:
            frames: list[np.ndarray] = []
            for _ in range(max(1, args.batch_size)):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            detections = detect_frames(
                model,
                frames,
                device=device,
                imgsz=args.imgsz,
                conf_thres=args.conf,
                min_conf=min(args.conf, args.track_low_conf),
                iou_thres=args.iou,
                max_det=args.max_det,
                max_nms_candidates=args.max_nms_candidates,
                names=names,
                keep_names=keep_names,
                half=use_half,
            )
            for frame, (boxes, scores, cls_ids, embeds) in zip(frames, detections):
                vis_boxes, vis_scores, vis_cls, vis_embeds, person_boxes, person_scores, person_embeds = visible_and_track_inputs(
                    boxes, scores, cls_ids, embeds, names, args.conf
                )
                person_tracks = tracker.update(person_boxes, person_scores, person_embeds)
                person_track_ids = assign_person_tracks_to_visible(
                    vis_boxes, vis_cls, names, person_tracks, args.track_assign_iou
                )
                track_ids, person_mask = match_parts_to_tracks(
                    model, vis_boxes, vis_scores, vis_cls, vis_embeds, names, person_track_ids, args.assoc_thres
                )
                track_ids = inherit_unmatched_parts_from_tracks(track_ids, vis_boxes, vis_cls, names, person_tracks)
                vis = draw_tracks(frame, vis_boxes, vis_scores, vis_cls, names, track_ids, person_mask, args.box_width)
                writer.write(vis)
                frame_idx += 1
                if frame_idx % 30 == 0:
                    suffix = f"/{total}" if total else ""
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    print(f"processed {frame_idx}{suffix} frames ({frame_idx / elapsed:.2f} fps)", flush=True)
    finally:
        cap.release()
        writer.release()
    print(f"saved {out_path} ({frame_idx} frames)")


if __name__ == "__main__":
    main()
