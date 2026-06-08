#!/usr/bin/env python3
"""YOLO11 训练入口：检测 loss + optimizer + checkpoint 保存。

标准 YOLO/COCO 目录 + data.yaml，例如::

    path: /data/coco
    train: images/train2017
    val: images/val2017
    names: {0: person, ...}

用法::

    python train.py --data coco.yaml --epochs 100 --batch 16 --scale n
    python train.py --data coco.yaml --weights yolo11n.pt --preview-only
"""

from __future__ import annotations

import argparse
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from dataloader.build import build_yolo_dataset, build_dataloader
from dataloader.hyp import TrainHyp
from dataloader.utils import load_data_yaml
from model import YOLO11L, YOLO11M, YOLO11N, YOLO11S, YOLO11X, load_pretrained_checkpoint, save_yolo11_ckpt
from model.loss import v8DetectionLoss
from model.train_hyp import ModelHyp
from model.val import validate
from utils.visualize import plot_training_batch

_SCALE_CLS = {"n": YOLO11N, "s": YOLO11S, "m": YOLO11M, "l": YOLO11L, "x": YOLO11X}


class ModelEMA:
    """Exponential Moving Average of model weights (port of Ultralytics ModelEMA).

    Keeps a shadow copy of every floating-point tensor in the model state_dict and
    updates it after each optimizer step. Saving the EMA weights (instead of the raw
    model) is what makes trained results match Ultralytics.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model: torch.nn.Module) -> None:
        if not self.enabled:
            return
        self.updates += 1
        d = self.decay(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model: torch.nn.Module, include=("names", "stride", "nc", "scale")) -> None:
        for k in include:
            if hasattr(model, k):
                setattr(self.ema, k, getattr(model, k))



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO11 detection training")
    p.add_argument("--data", type=str, required=True, help="data.yaml 路径")
    p.add_argument("--weights", "--weight", dest="weights", type=str, default="", help="预训练 .ckpt / .pt（可选）")
    p.add_argument("--scale", type=str, default="n", choices=tuple(_SCALE_CLS))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--project", type=str, default="runs/train")
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--save-period", type=int, default=10, help="每 N epoch 保存一次")
    p.add_argument("--fraction", type=float, default=1.0, help="训练集子集比例（调试用）")
    p.add_argument("--val-interval", type=int, default=1, help="每 N epoch 验证一次 mAP")
    p.add_argument("--patience", type=int, default=50, help="早停：mAP 连续 N epoch 不提升则停止（<=0 关闭）")
    p.add_argument("--val-conf", type=float, default=0.001, help="验证 NMS 置信度阈值")
    p.add_argument("--val-iou", type=float, default=0.7, help="验证 NMS IoU 阈值")
    p.add_argument("--val-max-det", type=int, default=300, help="验证每图最多检测框")
    p.add_argument("--val-fraction", type=float, default=1.0, help="验证集子集比例（大 val 集调试用）")
    p.add_argument("--val-log-interval", type=int, default=5, help="验证每 N 个 batch 打印一次进度")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-interval", type=int, default=1, help="每 N 个 batch 打印一次（默认 1=每 step）")
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--preview-batches", type=int, default=4)
    return p.parse_args()


def save_batch_preview(batch, names, out_dir: Path, tag: str) -> None:
    path = out_dir / f"{tag}.jpg"
    plot_training_batch(batch, names, path)
    print(f"saved sample grid -> {path.resolve()}")


def freeze_dfl(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if ".dfl" in name:
            param.requires_grad = False


def build_optimizer(model: torch.nn.Module, hyp: ModelHyp) -> torch.optim.SGD:
    g: list[list[torch.nn.Parameter]] = [[], [], []]
    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param_name == "bias":
                g[2].append(param)
            elif param_name == "weight" and isinstance(module, torch.nn.BatchNorm2d):
                g[1].append(param)
            else:
                g[0].append(param)
    return torch.optim.SGD(
        [
            {"params": g[0], "weight_decay": hyp.weight_decay, "initial_lr": hyp.lr0, "param_group": "decay"},
            {"params": g[1], "weight_decay": 0.0, "initial_lr": hyp.lr0, "param_group": "bn"},
            {"params": g[2], "weight_decay": 0.0, "initial_lr": hyp.lr0, "param_group": "bias"},
        ],
        lr=hyp.lr0,
        momentum=hyp.momentum,
        nesterov=True,
    )


def one_cycle(y1: float = 1.0, y2: float = 0.01, steps: int = 100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def adjust_lr(optimizer, ni: int, nw: int, epoch: int, lf, hyp: ModelHyp) -> None:
    lf_epoch = lf(epoch)
    if ni <= nw:
        xi = [0, nw]
        for pg in optimizer.param_groups:
            start = hyp.warmup_bias_lr if pg.get("param_group") == "bias" else 0.0
            end = pg["initial_lr"] * lf_epoch
            pg["lr"] = float(np.interp(ni, xi, [start, end]))
            if "momentum" in pg:
                pg["momentum"] = float(np.interp(ni, xi, [hyp.warmup_momentum, hyp.momentum]))
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lf_epoch


def batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def log_train_step(
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    loss_items: torch.Tensor,
    t_epoch: float,
) -> None:
    elapsed = time.time() - t_epoch
    eta = elapsed / max(step, 1) * (total_steps - step)
    total = loss_items.sum().item()
    print(
        f"[epoch {epoch + 1}/{total_epochs}] step {step}/{total_steps}  "
        f"box={loss_items[0].item():.4f} cls={loss_items[1].item():.4f} dfl={loss_items[2].item():.4f}  "
        f"loss={total:.4f}  elapsed={elapsed:.0f}s eta={eta:.0f}s",
        flush=True,
    )


def build_model(scale: str, nc: int, weights: str, device: torch.device) -> torch.nn.Module:
    model = _SCALE_CLS[scale](nc=nc).to(device)
    if weights:
        if not Path(weights).exists():
            raise FileNotFoundError(f"weights not found: {weights}")
        _, n_loaded, n_total = load_pretrained_checkpoint(weights, model, device="cpu", reinit_head=False)
        print(f"loaded pretrained: {n_loaded}/{n_total} tensors from {weights}")
    else:
        print("no pretrained weights provided, training from scratch")
    freeze_dfl(model)
    return model


def main() -> None:
    global args
    args = parse_args()
    hyp_aug = TrainHyp(
        imgsz=args.imgsz,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
    )
    hyp = ModelHyp(lr0=args.lr0, lrf=args.lrf)

    data = load_data_yaml(args.data)
    train_ds = build_yolo_dataset(
        data["train"],
        data,
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=True,
        hyp=hyp_aug,
        prefix="train: ",
        fraction=args.fraction,
    )
    val_ds = build_yolo_dataset(
        data["val"],
        data,
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        hyp=hyp_aug,
        prefix="val: ",
        fraction=args.val_fraction,
    )
    train_loader = build_dataloader(train_ds, args.batch, args.workers, shuffle=True, infinite=True)
    val_loader = build_dataloader(val_ds, args.batch, args.workers, shuffle=False, infinite=False)
    names: list[str] = list(data["names"])
    out_dir = Path(args.project) / args.name
    sample_dir = out_dir / "samples"
    weights_dir = out_dir / "weights"
    sample_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"train: {len(train_loader.dataset)} images")
    print(f"val:   {len(val_loader.dataset)} images")
    print(f"nc={data['nc']}  names={names}")

    n_preview = min(args.preview_batches, len(train_loader))
    train_iter = iter(train_loader)
    for i in range(n_preview):
        batch = next(train_iter)
        save_batch_preview(batch, names, sample_dir, f"train_aug_batch{i}")
    val_batch = next(iter(val_loader))
    save_batch_preview(val_batch, names, sample_dir, "val_letterbox_batch0")

    if args.preview_only:
        print("preview-only 完成。")
        return

    device = torch.device(args.device)
    model = build_model(args.scale, data["nc"], args.weights, device)
    model.args = hyp
    model.names = names
    criterion = v8DetectionLoss(model)
    accumulate = max(round(hyp.nbs / args.batch), 1)
    hyp.weight_decay *= args.batch * accumulate / hyp.nbs
    optimizer = build_optimizer(model, hyp)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    ema = ModelEMA(model)
    nb = len(train_loader)
    nw = max(round(hyp.warmup_epochs * nb), 100) if hyp.warmup_epochs > 0 else -1
    lf = one_cycle(1, hyp.lrf, args.epochs) if hyp.cos_lr else (lambda x: (1 - x / args.epochs) * (1.0 - hyp.lrf) + hyp.lrf)
    best_fitness = -1.0
    best_epoch = 0

    print(f"\nStart training: device={device} epochs={args.epochs} batch={args.batch} amp={scaler.is_enabled()}")
    print(f"  {nb} batches/epoch  (~{len(train_loader.dataset)} images)\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        if epoch == args.epochs - args.close_mosaic:
            train_loader.dataset.close_mosaic(hyp_aug)
            train_loader.reset()
            print(f"epoch {epoch}: mosaic/mixup/cutmix 已关闭")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_items_sum = torch.zeros(3, device=device)
        n_batches = 0
        train_iter = iter(train_loader)
        t_epoch = time.time()
        for i in range(nb):
            batch = batch_to_device(next(train_iter), device)
            imgs = batch["img"].float() / 255.0
            ni = i + nb * epoch
            adjust_lr(optimizer, ni, nw, epoch, lf, hyp)

            with autocast(enabled=scaler.is_enabled()):
                preds = model(imgs)
                loss, loss_items = criterion(preds, batch)

            scaler.scale(loss).backward()
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
            loss_items_sum += loss_items
            n_batches += 1

            step = i + 1
            if step % args.log_interval == 0 or step == nb:
                log_train_step(epoch, args.epochs, step, nb, loss_items, t_epoch)

            if epoch == 0 and i < 2:
                save_batch_preview(batch, names, sample_dir, f"epoch{epoch}_batch{i}")

        avg_items = loss_items_sum / max(n_batches, 1)
        total = avg_items.sum().item()
        dt = time.time() - t0
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"box={avg_items[0]:.4f} cls={avg_items[1]:.4f} dfl={avg_items[2]:.4f}  "
            f"loss={total:.4f}  time={dt:.1f}s"
        )

        last_path = weights_dir / "last.ckpt"
        ema.update_attr(model)
        save_yolo11_ckpt(ema.ema, last_path, meta={"epoch": epoch + 1, "names": names}, scale=args.scale, nc=data["nc"])

        run_val = args.val_interval > 0 and (
            (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs
        )
        if run_val:
            metrics = validate(
                ema.ema,
                val_loader,
                device,
                data["nc"],
                conf_thres=args.val_conf,
                iou_thres=args.val_iou,
                max_det=args.val_max_det,
                amp=scaler.is_enabled(),
                progress_interval=args.val_log_interval,
            )
            fitness = metrics.fitness
            print(
                f"           val: mAP50={metrics.map50:.4f} mAP50-95={metrics.map:.4f} "
                f"fitness={fitness:.4f} (best={max(best_fitness, 0.0):.4f}@ep{best_epoch})"
            )
            if fitness > best_fitness:
                best_fitness = fitness
                best_epoch = epoch + 1
                save_yolo11_ckpt(
                    ema.ema,
                    weights_dir / "best.ckpt",
                    meta={
                        "epoch": epoch + 1,
                        "names": names,
                        "fitness": best_fitness,
                        "map50": metrics.map50,
                        "map": metrics.map,
                    },
                    scale=args.scale,
                    nc=data["nc"],
                )
            if args.patience > 0 and (epoch + 1) - best_epoch >= args.patience:
                print(
                    f"早停：mAP 连续 {args.patience} epoch 未提升（best fitness={best_fitness:.4f} @ epoch {best_epoch}）"
                )
                break
        if args.save_period > 0 and (epoch + 1) % args.save_period == 0:
            save_yolo11_ckpt(
                ema.ema,
                weights_dir / f"epoch{epoch + 1}.ckpt",
                meta={"epoch": epoch + 1, "names": names},
                scale=args.scale,
                nc=data["nc"],
            )

    print(f"\nTraining done. weights -> {weights_dir.resolve()}")
    if best_fitness >= 0:
        print(f"best fitness={best_fitness:.4f} @ epoch {best_epoch}  ({weights_dir / 'best.ckpt'})")


if __name__ == "__main__":
    main()
