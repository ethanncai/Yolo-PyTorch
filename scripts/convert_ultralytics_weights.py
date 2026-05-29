#!/usr/bin/env python3
"""将 Ultralytics 发布的 ``yolo11*.pt`` 转成纯张量 ``.ckpt``（无自定义 pickle 类）。

用法::

    python scripts/convert_ultralytics_weights.py assets/ckpts/yolo11n.pt
    # 或（pip install -e . 后）: yolo11-convert-pt assets/ckpts/yolo11n.pt
    # 未指定 -o 时，默认写到 assets/ckpts/<basename>.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model.paths import ckpt_assets_dir
from model.utils import guess_scale_from_name
from model.weights import export_ultralytics_pt_to_ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description="Ultralytics .pt → torch-native .ckpt")
    ap.add_argument("src", type=Path, help="输入 yolo11*.pt")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出路径（默认：<project>/assets/ckpts/<输入基名>.ckpt）",
    )
    args = ap.parse_args()
    src: Path = args.src
    ckt_dir = ckpt_assets_dir()
    ckt_dir.mkdir(parents=True, exist_ok=True)
    dst: Path = (
        args.output if args.output is not None else ckt_dir / f"{src.stem}.ckpt"
    )
    scale = guess_scale_from_name(src)
    meta = {"variant": src.stem}
    if scale:
        meta["scale"] = scale
    export_ultralytics_pt_to_ckpt(src, dst, meta=meta)
    print(f"wrote {dst.resolve()}")


if __name__ == "__main__":
    main()
