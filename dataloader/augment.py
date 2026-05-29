"""YOLOv8/v11 训练增强（自 Ultralytics data/augment 精简，仅检测）。"""

from __future__ import annotations

import math
import random
from typing import Any

import cv2
import numpy as np
import torch

from .hyp import TrainHyp
from .instance import Instances
from .ops import bbox_ioa, xywh2xyxy


class BaseTransform:
    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        return {}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        return labels


class Compose:
    def __init__(self, transforms: list | BaseTransform):
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform: BaseTransform) -> None:
        self.transforms.append(transform)


class BaseMixTransform(BaseTransform):
    def __init__(self, dataset, pre_transform=None, p: float = 0.0) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if random.uniform(0, 1) > self.p:
            return labels
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels.pop("mix_labels", None)
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        if self.pre_transform is not None:
            mix_labels = [self.pre_transform(m) for m in mix_labels]
        labels["mix_labels"] = mix_labels
        return {"mix_labels": mix_labels}

    def get_indexes(self):
        return random.randint(0, len(self.dataset) - 1)


class Mosaic(BaseMixTransform):
    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        assert n in {4, 9}
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)
        self.n = n
        self.buffer_enabled = True

    def get_indexes(self):
        if self.buffer_enabled and self.dataset.buffer:
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        params = super().get_params(labels)
        assert len(labels.get("mix_labels", [])), "no mix labels for mosaic"
        s = self.imgsz
        layout = []
        if self.n == 4:
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
            for i in range(4):
                patch = labels if i == 0 else labels["mix_labels"][i - 1]
                h, w = patch.get("resized_shape", patch["img"].shape[:2])
                if i == 0:
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                else:
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                layout.append(
                    {
                        "labels_patch": patch,
                        "x1a": x1a, "y1a": y1a, "x2a": x2a, "y2a": y2a,
                        "x1b": x1b, "y1b": y1b, "x2b": x2b, "y2b": y2b,
                        "padw": x1a - x1b, "padh": y1a - y1b,
                        "img_shape": (h, w),
                    }
                )
        params["layout"] = layout
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        layout = params["layout"]
        c = labels["img"].shape[2]
        img4 = np.full((self.imgsz * 2, self.imgsz * 2, c), 114, dtype=np.uint8)
        for item in layout:
            img = item["labels_patch"]["img"]
            y1a, y2a, x1a, x2a = item["y1a"], item["y2a"], item["x1a"], item["x2a"]
            y1b, y2b, x1b, x2b = item["y1b"], item["y2b"], item["x1b"], item["x2b"]
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        labels["img"] = img4
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        mosaic_labels = []
        for item in params["layout"]:
            patch = self._update_labels(
                item["labels_patch"], item["padw"], item["padh"], item["img_shape"]
            )
            mosaic_labels.append(patch)
        final = self._cat_labels(mosaic_labels)
        labels.update(final)
        return labels

    @staticmethod
    def _update_labels(labels, padw, padh, img_shape):
        nh, nw = img_shape
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict]) -> dict:
        cls = [m["cls"] for m in mosaic_labels]
        instances = [m["instances"] for m in mosaic_labels]
        imgsz = self.imgsz * 2
        final = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
        }
        final["instances"].clip(imgsz, imgsz)
        good = final["instances"].remove_zero_area_boxes()
        final["cls"] = final["cls"][good]
        return final


class MixUp(BaseMixTransform):
    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        params = super().get_params(labels)
        params["r"] = np.random.beta(32.0, 32.0)
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        r = params["r"]
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        labels2 = labels["mix_labels"][0]
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class CutMix(BaseMixTransform):
    def __init__(self, dataset, pre_transform=None, p: float = 0.0, beta: float = 1.0, num_areas: int = 3):
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta = beta
        self.num_areas = num_areas

    def _rand_bbox(self, width: int, height: int):
        lam = np.random.beta(self.beta, self.beta)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
        cx, cy = np.random.randint(width), np.random.randint(height)
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        return x1, y1, x2, y2

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        params = super().get_params(labels)
        h, w = labels["img"].shape[:2]
        cut_areas = np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)
        if len(labels["instances"]):
            labels["instances"].convert_bbox("xyxy")
            labels["instances"].denormalize(w, h)
            ioa1 = bbox_ioa(cut_areas, labels["instances"].bboxes)
            idx = np.nonzero(ioa1.sum(axis=1) <= 0)[0]
        else:
            idx = np.arange(len(cut_areas))
        if len(idx) == 0:
            params["skip"] = True
            return params
        labels2 = labels["mix_labels"][0]
        area = cut_areas[np.random.choice(idx)]
        if len(labels2["instances"]):
            labels2["instances"].convert_bbox("xyxy")
            labels2["instances"].denormalize(w, h)
            ioa2 = bbox_ioa(area[None], labels2["instances"].bboxes).squeeze(0)
            indexes2 = np.nonzero(ioa2 >= 0.1)[0]
        else:
            indexes2 = np.array([], dtype=int)
        if len(indexes2) == 0:
            params["skip"] = True
            return params
        params.update({"area": area, "indexes2": indexes2, "w": w, "h": h})
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params.get("skip"):
            return labels
        x1, y1, x2, y2 = params["area"].astype(np.int32)
        labels2 = labels["mix_labels"][0]
        labels["img"][y1:y2, x1:x2] = labels2["img"][y1:y2, x1:x2]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        if params.get("skip"):
            return labels
        labels2 = labels["mix_labels"][0]
        w, h = params["w"], params["h"]
        x1, y1, x2, y2 = params["area"].astype(np.int32)
        indexes2 = params["indexes2"]
        instances2 = labels2["instances"][indexes2]
        instances2.convert_bbox("xyxy")
        instances2.denormalize(w, h)
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2 - x1, y2 - y1)
        instances2.add_padding(x1, y1)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"][indexes2]], axis=0)
        labels["instances"] = Instances.concatenate([labels["instances"], instances2], axis=0)
        return labels


class RandomPerspective(BaseTransform):
    def __init__(
        self,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 0.0,
        perspective: float = 0.0,
        size: tuple[int, int] | None = None,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.size = size

    def _compute_affine_matrix(self, img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
        C = np.eye(3, dtype=np.float32)
        C[0, 2], C[1, 2] = -img.shape[1] / 2, -img.shape[0] / 2
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[0]
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * size[1]
        return T @ S @ R @ P @ C, s

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels["img"]
        size = (img.shape[1], img.shape[0]) if self.size is None else self.size
        M, scale = self._compute_affine_matrix(img, size)
        return {"M": M, "scale": scale, "orig_shape": img.shape[:2], "size": size}

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        img = labels["img"]
        M, size = params["M"], params["size"]
        if (size[0], size[1]) != (img.shape[1], img.shape[0]) or (M != np.eye(3)).any():
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=size, borderValue=(114, 114, 114))
            else:
                img = cv2.warpAffine(img, M[:2], dsize=size, borderValue=(114, 114, 114))
        labels["img"] = img
        labels["resized_shape"] = (img.shape[0], img.shape[1])
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*params["orig_shape"][::-1])
        M, scale = params["M"], params["scale"]
        bboxes = self.apply_bboxes(instances.bboxes, M)
        new_instances = Instances(bboxes, bbox_format="xyxy", normalized=False)
        new_instances.clip(*params["size"])
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(instances.bboxes.T, new_instances.bboxes.T)
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        return labels

    def apply_bboxes(self, bboxes: np.ndarray, M: np.ndarray) -> np.ndarray:
        n = len(bboxes)
        if n == 0:
            return bboxes
        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)
        x, y = xy[:, [0, 2, 4, 6]], xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    @staticmethod
    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


class RandomHSV(BaseTransform):
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain, self.sgain, self.vgain = hgain, sgain, vgain

    def apply_image(self, labels, params=None):
        img = labels["img"]
        if img.shape[-1] != 3 or not (self.hgain or self.sgain or self.vgain):
            return labels
        dtype = img.dtype
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
        lut_sat[0] = 0
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return labels


class RandomFlip(BaseTransform):
    def __init__(self, p=0.5, direction="horizontal"):
        assert direction in {"horizontal", "vertical"}
        self.p, self.direction = p, direction

    def get_params(self, labels):
        img = labels["img"]
        instances = labels["instances"]
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w
        return {"flip": random.random() < self.p, "h": h, "w": w}

    def apply_image(self, labels, params):
        img = labels["img"]
        if params["flip"]:
            img = np.flipud(img) if self.direction == "vertical" else np.fliplr(img)
        labels["img"] = np.ascontiguousarray(img)
        return labels

    def apply_instances(self, labels, params):
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        if params["flip"]:
            if self.direction == "vertical":
                instances.flipud(params["h"])
            else:
                instances.fliplr(params["w"])
        labels["instances"] = instances
        return labels


class LetterBox(BaseTransform):
    def __init__(self, new_shape=(640, 640), scaleup=True, center=True, padding_value=114):
        self.new_shape = new_shape
        self.scaleup = scaleup
        self.center = center
        self.padding_value = padding_value

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        if image is not None:
            labels["img"] = image
        only_img = len(labels) == 1 and "img" in labels
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        if not only_img:
            labels = self.apply_instances(labels, params)
        return labels["img"] if only_img else labels

    def get_params(self, labels):
        img = labels["img"]
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        ratio = (r, r)
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.center:
            dw, dh = dw / 2, dh / 2
        top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
        left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
        return {
            "orig_shape": shape, "new_shape": new_shape, "ratio": ratio,
            "new_unpad": new_unpad, "top": top, "bottom": bottom, "left": left, "right": right,
        }

    def apply_image(self, labels, params):
        img = labels["img"]
        if img.shape[:2][::-1] != params["new_unpad"]:
            img = cv2.resize(img, params["new_unpad"], interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        labels["img"] = cv2.copyMakeBorder(
            img, params["top"], params["bottom"], params["left"], params["right"],
            cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3,
        )
        labels["resized_shape"] = params["new_shape"]
        return labels

    def apply_instances(self, labels, params):
        if "instances" not in labels:
            return labels
        inst = labels["instances"]
        inst.convert_bbox(format="xyxy")
        inst.denormalize(*params["orig_shape"][::-1])
        inst.scale(*params["ratio"])
        inst.add_padding(params["left"], params["top"])
        return labels


class Format(BaseTransform):
    def __init__(self, bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.batch_idx = batch_idx
        self.bgr = bgr

    def get_params(self, labels):
        img = labels.get("img")
        h, w = img.shape[:2] if img is not None else (0, 0)
        cls = labels.pop("cls", np.array([]))
        instances = labels.pop("instances", None)
        if instances is not None:
            instances.convert_bbox(format=self.bbox_format)
            instances.denormalize(w, h)
        return {"h": h, "w": w, "cls": cls, "instances": instances, "nl": len(instances) if instances else 0}

    def apply_image(self, labels, params=None):
        img = labels.pop("img", None)
        if img is not None:
            if img.ndim == 2:
                img = img[..., None]
            img = img.transpose(2, 0, 1)
            if random.uniform(0, 1) > self.bgr and img.shape[0] == 3:
                img = img[::-1]
            labels["img"] = torch.from_numpy(np.ascontiguousarray(img))
        return labels

    def apply_instances(self, labels, params=None):
        cls = params["cls"]
        instances = params["instances"]
        nl = params["nl"]
        w, h = params["w"], params["h"]
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros((0, 1))
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((0, 4))
        if self.normalize and nl:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels


def v8_transforms(dataset, imgsz: int, hyp: TrainHyp) -> Compose:
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        size=(imgsz, imgsz),
    )
    pre_transform = Compose([mosaic, affine])
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr),
        ]
    )
