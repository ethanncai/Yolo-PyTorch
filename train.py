#!/usr/bin/env python3
"""YOLO11 训练入口：数据加载、增强可视化（loss 训练循环待接入）。

标准 YOLO/COCO 目录 + data.yaml，例如::

    path: /data/coco
    train: images/train2017
    val: images/val2017
    names: {0: person, ...}

用法::

    python train.py --data coco.yaml --epochs 3 --batch 8
    python train.py --data coco.yaml --preview-only   # 仅预览增强样本
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dataloader import TrainHyp, build_train_val_loaders
from utils.visualize import plot_training_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11 training (dataloader + aug preview)")
    p.add_argument("--data", type=str, required=True, help="data.yaml 路径")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--mosaic", type=float, default=1.0, help="mosaic 概率")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--close-mosaic", type=int, default=10, help="最后 N epoch 关闭 mosaic")
    p.add_argument("--preview-only", action="store_true", help="只跑若干 batch 并保存可视化")
    p.add_argument("--preview-batches", type=int, default=4)
    return p.parse_args()


def save_batch_preview(batch, names, out_dir: Path, tag: str) -> None:
    path = out_dir / f"{tag}.jpg"
    plot_training_batch(batch, names, path)
    print(f"saved sample grid -> {path.resolve()}")


def main() -> None:
    args = parse_args()
    hyp = TrainHyp(
        imgsz=args.imgsz,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
    )
    train_loader, val_loader, data = build_train_val_loaders(
        args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        hyp=hyp,
    )
    names: list[str] = list(data["names"])
    out_dir = Path(args.project) / args.name / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"train: {len(train_loader.dataset)} images")
    print(f"val:   {len(val_loader.dataset)} images")
    print(f"nc={data['nc']}  names={names[:5]}{'…' if len(names) > 5 else ''}")

    # 训练集增强预览
    n_preview = min(args.preview_batches, len(train_loader))
    train_iter = iter(train_loader)
    for i in range(n_preview):
        batch = next(train_iter)
        save_batch_preview(batch, names, out_dir, f"train_aug_batch{i}")

    # 验证集（LetterBox，无 mosaic）
    val_batch = next(iter(val_loader))
    save_batch_preview(val_batch, names, out_dir, "val_letterbox_batch0")

    if args.preview_only:
        print("preview-only 完成。")
        return

    device = torch.device(args.device)
    print(f"\n注意：检测 loss / optimizer 尚未实现，当前仅演示 dataloader 迭代。")
    print(f"device={device}  epochs={args.epochs}  close_mosaic={args.close_mosaic}\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        if epoch == args.epochs - args.close_mosaic:
            train_loader.dataset.close_mosaic(hyp)
            train_loader.reset()
            print(f"epoch {epoch}: mosaic/mixup/cutmix 已关闭")

        n_batches = 0
        n_objects = 0
        train_iter = iter(train_loader)
        for _ in range(len(train_loader)):
            batch = next(train_iter)
            imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
            n_batches += 1
            n_objects += batch["cls"].shape[0]
            if n_batches <= 2 and epoch == 0:
                save_batch_preview(batch, names, out_dir, f"epoch{epoch}_batch{n_batches - 1}")

        dt = time.time() - t0
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"batches={n_batches}  objects={n_objects}  time={dt:.1f}s"
        )


if __name__ == "__main__":
    main()
