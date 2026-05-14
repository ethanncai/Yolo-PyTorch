"""项目内资源路径（依赖可安装的 ``model`` 包位置；开发时用 ``pip install -e .``）。"""

from __future__ import annotations

from pathlib import Path

_MODEL_PKG = Path(__file__).resolve().parent


def model_package_dir() -> Path:
    """本包目录（…/model）。"""
    return _MODEL_PKG


def project_root() -> Path:
    """项目根目录（与 ``model`` 并列，含 ``assets/``、``pyproject.toml``）。"""
    return _MODEL_PKG.parent


def assets_dir() -> Path:
    return project_root() / "assets"


def ckpt_assets_dir() -> Path:
    """预置/转换后的 ``*.ckpt`` / 可选 ``*.pt`` 源权重目录：``assets/ckpts``。"""
    return assets_dir() / "ckpts"
