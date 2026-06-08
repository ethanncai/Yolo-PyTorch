"""Torch-native checkpoints (``state_dict`` only) + legacy ``.pt`` without Ultralytics runtime."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "CKPT_FORMAT_VERSION",
    "intersect_dicts",
    "load_yolo11_checkpoint",
    "load_pretrained_checkpoint",
    "load_checkpoint_file",
    "save_yolo11_ckpt",
    "export_ultralytics_pt_to_ckpt",
]

CKPT_FORMAT_VERSION = 1


class _LegacyPtUnpickler(pickle.Unpickler):
    """Unpickle Ultralytics ``.pt`` by mapping ``ultralytics.*`` classes to ``nn.Module`` stubs."""

    def find_class(self, module: str, name: str):
        if module.startswith("ultralytics"):
            return nn.Module
        return super().find_class(module, name)


class _LegacyPtPickle:
    Unpickler = _LegacyPtUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def load_checkpoint_file(path: str | Path, *, device: str | torch.device | None = None) -> dict[str, Any]:
    """Load checkpoint dict from native ``.ckpt`` or legacy Ultralytics ``.pt`` (no ultralytics import)."""
    path = Path(path)
    kwargs: dict[str, Any] = {"map_location": device or "cpu", "weights_only": False}
    if path.suffix.lower() == ".pt":
        kwargs["pickle_module"] = _LegacyPtPickle
    ckpt = torch.load(path, **kwargs)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected dict checkpoint, got {type(ckpt)}")
    return ckpt


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


def intersect_dicts(da: dict[str, torch.Tensor], db: dict[str, torch.Tensor], exclude: tuple = ()) -> dict[str, torch.Tensor]:
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


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
    - Legacy Ultralytics ``.pt``: unpickled via stub loader (no ``ultralytics`` package)
    """
    ckpt = load_checkpoint_file(path, device=device)
    sd = _remap_state_dict_keys(_extract_raw_state_dict(ckpt))
    model.load_state_dict(sd, strict=strict)
    return ckpt


def load_pretrained_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    device: str | torch.device | None = None,
    reinit_head: bool = False,
) -> tuple[dict[str, Any], int, int]:
    """Load matching weights only (e.g. COCO pretrain -> custom nc)."""
    ckpt = load_checkpoint_file(path, device=device)
    sd = _remap_state_dict_keys(_extract_raw_state_dict(ckpt))
    current = model.state_dict()
    filtered = intersect_dicts(sd, current)
    model.load_state_dict(filtered, strict=False)
    if reinit_head:
        head = model.model[-1]
        if hasattr(head, "bias_init"):
            head.bias_init()
    return ckpt, len(filtered), len(current)


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
    One-shot: read legacy ``yolo11*.pt``, write native ``.ckpt`` with raw tensors only.
    Does not require the ``ultralytics`` package at runtime.
    """
    ultralytics_pt, out_ckpt = Path(ultralytics_pt), Path(out_ckpt)
    ckpt = load_checkpoint_file(ultralytics_pt, device="cpu")
    sd = _remap_state_dict_keys(_extract_raw_state_dict(ckpt))
    extra = dict(meta or {})
    extra.setdefault("source_pt", str(ultralytics_pt.resolve()))
    out = {"format_version": CKPT_FORMAT_VERSION, "meta": extra, "state_dict": sd}
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_ckpt)
    return out
