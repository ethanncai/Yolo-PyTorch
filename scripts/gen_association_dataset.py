#!/usr/bin/env python3
"""Generate a YOLO dataset with offline part-to-person association labels.

The input is a standard YOLO detection dataset. The output keeps a standard
YOLO layout, but each label row is extended from:

    cls x y w h

to:

    cls x y w h person_id

``person_id`` is image-local. ``-1`` means the object is not confidently
associated with any person/group.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm


IMG_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_PERSON_NAMES = ("person", "body")
DEFAULT_FACE_NAMES = ("face", "head")
DEFAULT_HAND_NAMES = ("hand", "left_hand", "right_hand")


@dataclass
class YoloObject:
    cls: int
    x: float
    y: float
    w: float
    h: float
    person_id: int = -1

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.w / 2,
            self.y - self.h / 2,
            self.x + self.w / 2,
            self.y + self.h / 2,
        )

    @property
    def center(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class PosePerson:
    xyxy: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float]]


def parse_name_set(value: str) -> set[str]:
    return {x.strip().lower() for x in value.split(",") if x.strip()}


def load_dataset_yaml(src: Path) -> tuple[Path, dict[str, Any]]:
    src = src.expanduser().resolve()
    if src.is_dir():
        candidates = sorted(src.glob("*.yaml")) + sorted(src.glob("*.yml"))
        if not candidates:
            raise FileNotFoundError(f"no data.yaml found in {src}")
        yaml_path = candidates[0]
    else:
        yaml_path = src
    with yaml_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid dataset yaml: {yaml_path}")
    data["yaml_file"] = str(yaml_path)
    return yaml_path, data


def normalize_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if names is None:
        nc = int(data.get("nc", 0))
        return [f"class_{i}" for i in range(nc)]
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    return [str(x) for x in names]


def resolve_dataset_path(root: Path, value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        path = Path(item).expanduser()
        paths.append(path if path.is_absolute() else (root / path).resolve())
    return paths


def is_image_candidate(path: Path) -> bool:
    name = path.name
    if path.suffix.lower() not in IMG_SUFFIXES:
        return False
    if name.startswith(".") or name.startswith("._") or "_fftmp" in name:
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def image_entries(paths: list[Path]) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    for path in paths:
        if path.is_dir():
            for img in sorted(p for p in path.rglob("*") if is_image_candidate(p)):
                entries.append((img, path))
        elif path.is_file():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    img = Path(line.strip()).expanduser()
                    if not img:
                        continue
                    if not img.is_absolute():
                        img = (path.parent / img).resolve()
                    if is_image_candidate(img):
                        entries.append((img, img.parent))
        else:
            raise FileNotFoundError(path)
    return entries


def label_path_for_image(img: Path) -> Path:
    parts = list(img.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img.parent.parent / "labels" / img.with_suffix(".txt").name


def read_label_file(path: Path) -> list[YoloObject]:
    if not path.is_file():
        return []
    objects: list[YoloObject] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            cols = raw.split()
            if len(cols) < 5:
                raise ValueError(f"{path}:{line_no}: expected at least 5 columns")
            cls, x, y, w, h = cols[:5]
            person_id = int(float(cols[5])) if len(cols) >= 6 else -1
            objects.append(YoloObject(int(float(cls)), float(x), float(y), float(w), float(h), person_id))
    return objects


def point_in_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def expand_box(box: tuple[float, float, float, float], ratio: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return x1 - w * ratio, y1 - h * ratio, x2 + w * ratio, y2 + h * ratio


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def visible_points(
    person: PosePerson,
    indices: tuple[int, ...],
    min_conf: float,
) -> list[tuple[float, float]]:
    points = []
    for idx in indices:
        if idx >= len(person.keypoints):
            continue
        x, y, conf = person.keypoints[idx]
        if conf >= min_conf and x > 0 and y > 0:
            points.append((x, y))
    return points


def mean_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)


def points_inside_ratio(points: list[tuple[float, float]], box: tuple[float, float, float, float]) -> float:
    if not points:
        return 0.0
    return sum(point_in_box(point, box) for point in points) / len(points)


def load_pose_model(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("需要安装 ultralytics 才能使用 yolo11m-pose：pip install ultralytics") from exc
    return YOLO(weights)


def read_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def predict_pose_people(
    pose_model,
    img_path: Path,
    *,
    imgsz: int,
    conf: float,
    device: str | None,
    kpt_conf: float,
) -> list[PosePerson]:
    kwargs: dict[str, Any] = {"verbose": False, "conf": conf}
    if imgsz > 0:
        kwargs["imgsz"] = imgsz
    if device:
        kwargs["device"] = device
    image = read_rgb_image(img_path)
    result = pose_model.predict(image, **kwargs)[0]
    orig_h, orig_w = result.orig_shape
    if result.keypoints is None or result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    kpts_xy = result.keypoints.xy.detach().cpu().tolist()
    kpts_conf = result.keypoints.conf.detach().cpu().tolist() if result.keypoints.conf is not None else []
    people: list[PosePerson] = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        keypoints = []
        for j, point in enumerate(kpts_xy[i]):
            conf_j = kpts_conf[i][j] if kpts_conf else kpt_conf
            keypoints.append((point[0] / orig_w, point[1] / orig_h, conf_j))
        people.append(
            PosePerson(
                xyxy=(x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h),
                keypoints=keypoints,
            )
        )
    return people


def object_kind(obj: YoloObject, names: list[str], person_names: set[str], face_names: set[str], hand_names: set[str]) -> str:
    lower_names = [name.lower() for name in names]
    name = lower_names[obj.cls] if 0 <= obj.cls < len(lower_names) else str(obj.cls)
    if name in person_names:
        return "person"
    if name in face_names:
        return "face"
    if name in hand_names:
        return "hand"
    return "other"


def score_pose_object(obj: YoloObject, obj_kind: str, pose: PosePerson, kpt_conf: float) -> float:
    obj_box = obj.xyxy
    pose_box = pose.xyxy
    pose_w = max(pose_box[2] - pose_box[0], 1e-6)
    pose_h = max(pose_box[3] - pose_box[1], 1e-6)
    pose_diag = (pose_w * pose_w + pose_h * pose_h) ** 0.5

    if obj_kind == "person":
        torso = visible_points(pose, (5, 6, 11, 12), kpt_conf)
        return box_iou(obj_box, pose_box) * 3.0 + points_inside_ratio(torso, obj_box)

    if obj_kind == "face":
        # head_anchor = mean of visible COCO head keypoints (nose/eyes/ears),
        # mirroring cosa-cv's identity_assigner.head_anchor_from_kpts. The hard
        # gate "anchor must fall inside the face box" disambiguates two people
        # standing close (their pose boxes overlap heavily, but each head
        # anchor still lands cleanly inside its own face box).
        head = visible_points(pose, (0, 1, 2, 3, 4), kpt_conf)
        anchor = mean_point(head)
        if anchor is not None:
            if not point_in_box(anchor, obj_box):
                # Head anchor known but not inside this face box -> reject.
                return -999.0
            # Inside: closer anchor-to-face-center => higher score.
            face_w = max(obj_box[2] - obj_box[0], 1e-6)
            face_h = max(obj_box[3] - obj_box[1], 1e-6)
            face_diag = (face_w * face_w + face_h * face_h) ** 0.5
            dist_score = 1.0 - distance(obj.center, anchor) / max(face_diag, 1e-6)
            return 2.0 + dist_score + points_inside_ratio(head, obj_box) * 2.0
        # No usable head keypoint: fall back to the pose-bbox head estimate.
        center = ((pose_box[0] + pose_box[2]) / 2, pose_box[1] + pose_h * 0.16)
        dist_score = 1.0 - distance(obj.center, center) / max(pose_diag * 0.25, 1e-6)
        return dist_score

    if obj_kind == "hand":
        wrists = visible_points(pose, (9, 10), kpt_conf)
        arms = visible_points(pose, (7, 8, 9, 10), kpt_conf)
        candidates = wrists or arms
        if not candidates:
            return -999.0
        nearest = min(distance(obj.center, point) for point in candidates)
        dist_score = 1.0 - nearest / max(pose_diag * 0.35, 1e-6)
        return dist_score + points_inside_ratio(wrists, obj_box) * 2.0

    return -999.0


def _hungarian_threshold(score_mat: np.ndarray, min_score: float) -> list[tuple[int, int]]:
    """1-to-1 assignment via Hungarian algorithm; only return pairs with score >= min_score."""
    if score_mat.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-score_mat)
    except ImportError:
        # Greedy unique fallback when scipy is unavailable.
        row_ind, col_ind = [], []
        taken: set[int] = set()
        for r in np.argsort(-score_mat.max(axis=1)):
            avail = [c for c in range(score_mat.shape[1]) if c not in taken]
            if not avail:
                break
            best_c = max(avail, key=lambda c: score_mat[r, c])
            row_ind.append(int(r))
            col_ind.append(best_c)
            taken.add(best_c)
    return [
        (int(r), int(c))
        for r, c in zip(row_ind, col_ind)
        if float(score_mat[r, c]) >= min_score
    ]


def _capacity_hungarian(
    score_mat: np.ndarray, min_score: float, capacity: int
) -> list[tuple[int, int]]:
    """Hungarian assignment allowing up to `capacity` rows assigned to each column."""
    if capacity <= 1:
        return _hungarian_threshold(score_mat, min_score)
    # Replicate each column `capacity` times to turn it into a 1-to-1 problem.
    expanded = np.tile(score_mat, capacity)  # (n_rows, n_cols * capacity)
    pairs = _hungarian_threshold(expanded, min_score)
    n_refs = score_mat.shape[1]
    return [(r, c % n_refs) for r, c in pairs]


def associate_objects_with_pose(
    objects: list[YoloObject],
    names: list[str],
    person_names: set[str],
    face_names: set[str],
    hand_names: set[str],
    pose_people: list[PosePerson],
    kpt_conf: float,
    min_score: float,
) -> bool:
    if not pose_people:
        return False

    # Group objects by kind.
    by_kind: dict[str, list[int]] = {"person": [], "face": [], "hand": []}
    for i, obj in enumerate(objects):
        k = object_kind(obj, names, person_names, face_names, hand_names)
        if k in by_kind:
            by_kind[k].append(i)

    associated = False
    # persons/faces: strict 1-to-1 per pose; hands: up to 2 per pose.
    for kind_str, capacity in (("person", 1), ("face", 1), ("hand", 2)):
        idxs = by_kind[kind_str]
        if not idxs:
            continue
        score_mat = np.array(
            [
                [score_pose_object(objects[oi], kind_str, pose, kpt_conf) for pose in pose_people]
                for oi in idxs
            ],
            dtype=float,
        )
        pairs = (
            _hungarian_threshold(score_mat, min_score)
            if capacity == 1
            else _capacity_hungarian(score_mat, min_score, capacity)
        )
        for row, col in pairs:
            objects[idxs[row]].person_id = col
            associated = True

    return associated


def score_person_part(person: YoloObject, part: YoloObject, kind: str) -> float:
    px1, py1, px2, py2 = person.xyxy
    pw, ph = max(px2 - px1, 1e-6), max(py2 - py1, 1e-6)
    pcx, pcy = person.center
    x, y = part.center
    normalized_dist = distance((x, y), (pcx, pcy)) / ((pw * pw + ph * ph) ** 0.5)
    score = -normalized_dist
    if point_in_box(part.center, person.xyxy):
        score += 2.0
    if kind == "face":
        # Faces should usually sit in the upper body area.
        score += max(0.0, 1.0 - abs((y - py1) / ph - 0.18) * 3.0)
    elif kind == "hand":
        # Hands can be outside the box, but are usually close to the torso.
        score += max(0.0, 1.0 - abs((y - py1) / ph - 0.55) * 1.5)
    return score


def associate_objects(
    objects: list[YoloObject],
    names: list[str],
    person_names: set[str],
    face_names: set[str],
    hand_names: set[str],
    face_hand_max_dist: float,
) -> None:
    def kind(obj: YoloObject) -> str:
        return object_kind(obj, names, person_names, face_names, hand_names)

    people = [i for i, obj in enumerate(objects) if kind(obj) == "person"]
    faces = [i for i, obj in enumerate(objects) if kind(obj) == "face"]
    hands = [i for i, obj in enumerate(objects) if kind(obj) == "hand"]

    for pid, idx in enumerate(people):
        objects[idx].person_id = pid

    if people:
        person_ids = [objects[pidx].person_id for pidx in people]
        for kind_key, part_idxs, capacity in (("face", faces, 1), ("hand", hands, 2)):
            if not part_idxs:
                continue
            expand_ratio = 0.12 if kind_key == "face" else 0.28
            # Cells where the part center is outside the expanded person box stay at
            # -999.0, effectively excluded by min_score = -998.0.
            score_mat = np.full((len(part_idxs), len(people)), -999.0)
            for row, pidx in enumerate(part_idxs):
                for col, person_idx in enumerate(people):
                    person = objects[person_idx]
                    if point_in_box(objects[pidx].center, expand_box(person.xyxy, expand_ratio)):
                        score_mat[row, col] = score_person_part(person, objects[pidx], kind_key)
            pairs = (
                _hungarian_threshold(score_mat, -998.0)
                if capacity == 1
                else _capacity_hungarian(score_mat, -998.0, capacity)
            )
            for row, col in pairs:
                objects[part_idxs[row]].person_id = person_ids[col]
        return

    # If no body/person is visible, build weaker face-hand groups by proximity.
    next_pid = 0
    for idx in faces:
        objects[idx].person_id = next_pid
        next_pid += 1
    for hidx in hands:
        if not faces:
            continue
        nearest_face = min(faces, key=lambda fidx: distance(objects[hidx].center, objects[fidx].center))
        if distance(objects[hidx].center, objects[nearest_face].center) <= face_hand_max_dist:
            objects[hidx].person_id = objects[nearest_face].person_id


def write_label_file(path: Path, objects: list[YoloObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objects:
            f.write(f"{obj.cls} {obj.x:.6f} {obj.y:.6f} {obj.w:.6f} {obj.h:.6f} {obj.person_id}\n")


def copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode in {"auto", "hardlink"}:
        try:
            os.link(src, dst)
            return
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(src, dst)


def draw_viz(img_path: Path, label_path: Path, names: list[str], out_path: Path) -> None:
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    colors = [
        (239, 83, 80),
        (66, 165, 245),
        (102, 187, 106),
        (255, 202, 40),
        (171, 71, 188),
        (38, 198, 218),
        (255, 112, 67),
    ]
    for obj in read_label_file(label_path):
        x1, y1, x2, y2 = obj.xyxy
        box = (x1 * w, y1 * h, x2 * w, y2 * h)
        color = colors[obj.person_id % len(colors)] if obj.person_id >= 0 else (180, 180, 180)
        name = names[obj.cls] if 0 <= obj.cls < len(names) else str(obj.cls)
        text = f"{name} id={obj.person_id}"
        thick = 3
        for d in range(thick):
            draw.rectangle(
                (box[0] + d, box[1] + d, box[2] - d, box[3] - d),
                outline=color,
            )
        draw.text((box[0] + 2, max(0, box[1] - 12)), text, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def relative_image_path(img: Path, root: Path) -> Path:
    try:
        return img.relative_to(root)
    except ValueError:
        return Path(img.name)


def build_output_yaml(src_data: dict[str, Any], names: list[str], splits: dict[str, bool]) -> dict[str, Any]:
    out_data: dict[str, Any] = {"path": ".", "train": "images/train", "val": "images/val", "names": names}
    if "channels" in src_data:
        out_data["channels"] = src_data["channels"]
    for split in ("test",):
        if splits.get(split):
            out_data[split] = f"images/{split}"
    return out_data


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate YOLO labels with image-local person association ids.")
    ap.add_argument("--src", type=Path, required=True, help="标准 YOLO 数据集目录或 data.yaml")
    ap.add_argument("--out", type=Path, required=True, help="输出数据集目录")
    ap.add_argument("--fraction", type=float, default=1.0, help="每个 split 使用的数据比例，默认 1.0")
    ap.add_argument("--seed", type=int, default=0, help="抽样和可视化随机种子")
    ap.add_argument("--viz", type=int, default=8, help="生成多少张可视化样例，默认 8")
    ap.add_argument("--copy-mode", choices=("auto", "copy", "hardlink", "symlink"), default="auto")
    ap.add_argument("--overwrite", action="store_true", help="允许删除并重建已存在的输出目录")
    ap.add_argument("--pose-weights", default="yolo11m-pose.pt", help="YOLO pose 权重，默认 yolo11m-pose.pt")
    ap.add_argument("--pose-imgsz", type=int, default=640, help="pose 推理尺寸，默认 640")
    ap.add_argument("--pose-conf", type=float, default=0.25, help="pose person 置信度阈值")
    ap.add_argument("--pose-kpt-conf", type=float, default=0.25, help="关键点可见置信度阈值")
    ap.add_argument("--pose-min-score", type=float, default=0.0, help="框与 pose 匹配的最低分")
    ap.add_argument("--device", default=None, help="pose 推理设备，例如 0/cuda:0/cpu")
    ap.add_argument("--person-names", default=",".join(DEFAULT_PERSON_NAMES), help="person/body 类名，逗号分隔")
    ap.add_argument("--face-names", default=",".join(DEFAULT_FACE_NAMES), help="face/head 类名，逗号分隔")
    ap.add_argument("--hand-names", default=",".join(DEFAULT_HAND_NAMES), help="hand 类名，逗号分隔")
    ap.add_argument("--face-hand-max-dist", type=float, default=0.45, help="无 person 时 face-hand 最大中心距离")
    args = ap.parse_args()

    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")
    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.out} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    yaml_path, data = load_dataset_yaml(args.src)
    root = Path(data.get("path") or yaml_path.parent).expanduser()
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    names = normalize_names(data)

    rng = random.Random(args.seed)
    person_names = parse_name_set(args.person_names)
    face_names = parse_name_set(args.face_names)
    hand_names = parse_name_set(args.hand_names)
    pose_model = load_pose_model(args.pose_weights)

    processed: list[tuple[Path, Path]] = []
    split_has_data: dict[str, bool] = {}
    total_images = 0
    total_objects = 0
    total_associated = 0
    pose_hits = 0
    fallback_hits = 0
    skipped_images = 0

    for split in ("train", "val", "test"):
        if not data.get(split):
            continue
        split_paths = resolve_dataset_path(root, data[split])
        entries = image_entries(split_paths)
        if args.fraction < 1.0:
            rng.shuffle(entries)
            keep = max(1, int(round(len(entries) * args.fraction))) if entries else 0
            entries = sorted(entries[:keep], key=lambda x: str(x[0]))
        split_has_data[split] = bool(entries)
        for img_path, split_root in tqdm(entries, desc=f"{split}: generate"):
            rel = relative_image_path(img_path, split_root)
            out_img = args.out / "images" / split / rel
            out_label = args.out / "labels" / split / rel.with_suffix(".txt")
            try:
                objects = read_label_file(label_path_for_image(img_path))
                pose_people = predict_pose_people(
                    pose_model,
                    img_path,
                    imgsz=args.pose_imgsz,
                    conf=args.pose_conf,
                    device=args.device,
                    kpt_conf=args.pose_kpt_conf,
                )
                used_pose = associate_objects_with_pose(
                    objects,
                    names,
                    person_names,
                    face_names,
                    hand_names,
                    pose_people,
                    args.pose_kpt_conf,
                    args.pose_min_score,
                )
                if used_pose:
                    pose_hits += 1
                else:
                    associate_objects(
                        objects,
                        names,
                        person_names,
                        face_names,
                        hand_names,
                        args.face_hand_max_dist,
                    )
                    fallback_hits += 1
                copy_image(img_path, out_img, args.copy_mode)
                write_label_file(out_label, objects)
            except (OSError, ValueError) as exc:
                skipped_images += 1
                tqdm.write(f"WARNING: skip unreadable/corrupt sample {img_path}: {exc}")
                continue
            processed.append((out_img, out_label))
            total_images += 1
            total_objects += len(objects)
            total_associated += sum(obj.person_id >= 0 for obj in objects)

    out_yaml = build_output_yaml(data, names, split_has_data)
    with (args.out / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False, allow_unicode=True)

    if args.viz > 0 and processed:
        viz_samples = rng.sample(processed, k=min(args.viz, len(processed)))
        for i, (img_path, label_path) in enumerate(viz_samples):
            draw_viz(img_path, label_path, names, args.out / "viz" / f"sample_{i:02d}.jpg")

    print(f"wrote dataset: {args.out.resolve()}")
    print(f"images={total_images} objects={total_objects} associated={total_associated}")
    print(f"pose_associated_images={pose_hits} fallback_images={fallback_hits} skipped_images={skipped_images}")
    print(f"viz={min(args.viz, len(processed)) if processed else 0} -> {args.out / 'viz'}")


if __name__ == "__main__":
    main()
