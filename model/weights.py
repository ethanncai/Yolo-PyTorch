"""Torch-native checkpoints (``state_dict`` only) + optional import from legacy ``.pt``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "CKPT_FORMAT_VERSION",
    "load_yolo11_checkpoint",
    "save_yolo11_ckpt",
    "export_ultralytics_pt_to_ckpt",
]

CKPT_FORMAT_VERSION = 1


def _remap_state_dict_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        for old, new in (("model.model.", "model."), ("module.", "")):
            if nk.startswith(old):
                nk = new + nk[len(old) :]
        out[nk] = v
    return out


def _extract_raw_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    inner = ckpt.get("model")
    if inner is None:
        raise KeyError("Checkpoint has neither 'state_dict' nor 'model' (legacy Ultralytics .pt)")
    sd = inner.state_dict() if hasattr(inner, "state_dict") else inner
    if not isinstance(sd, dict):
        raise TypeError("Could not obtain a state_dict from checkpoint")
    return sd


def load_yolo11_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load weights into ``model``. Supports:

    - Native ``.ckpt``: ``{ "format_version", "meta", "state_dict" }``
    - Legacy Ultralytics ``.pt``: ``{ "model": nn.Module | state_dict, ... }``
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(ckpt)}")
    sd = _remap_state_dict_keys(_extract_raw_state_dict(ckpt))
    model.load_state_dict(sd, strict=strict)
    return ckpt


def save_yolo11_ckpt(
    model: nn.Module,
    path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
    scale: str | None = None,
    nc: int | None = None,
) -> None:
    """Save only tensors + small metadata (no pickled custom classes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m: dict[str, Any] = dict(meta or {})
    if scale is None and hasattr(model, "scale"):
        scale = getattr(model, "scale")
    if nc is None and hasattr(model, "nc"):
        nc = int(getattr(model, "nc"))
    if scale is not None:
        m["scale"] = scale
    if nc is not None:
        m["nc"] = nc
    payload = {
        "format_version": CKPT_FORMAT_VERSION,
        "meta": m,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def export_ultralytics_pt_to_ckpt(
    ultralytics_pt: str | Path,
    out_ckpt: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    One-shot: read legacy ``yolo11*.pt`` (pickle may contain ``DetectionModel``), write ``.ckpt`` with raw tensors.
    Does not import ``ultralytics``; uses ``torch.load`` only.
    """
    ultralytics_pt, out_ckpt = Path(ultralytics_pt), Path(out_ckpt)
    ckpt = torch.load(ultralytics_pt, map_location="cpu", weights_only=False)
    sd = _remap_state_dict_keys(_extract_raw_state_dict(ckpt))
    extra = dict(meta or {})
    extra.setdefault("source_pt", str(ultralytics_pt.resolve()))
    out = {"format_version": CKPT_FORMAT_VERSION, "meta": extra, "state_dict": sd}
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_ckpt)
    return out
