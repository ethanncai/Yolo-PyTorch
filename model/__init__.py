"""
YOLO11 检测：显式 ``ModuleList`` + ``forward``，标准 ``.ckpt`` / 旧 ``.pt`` 权重加载。

安装（可编辑模式，便于跨模块导入）::

    pip install -e .

示例::

    from model import YOLO11N, load_yolo11_checkpoint
    from model.paths import ckpt_assets_dir

    m = YOLO11N()
    load_yolo11_checkpoint(ckpt_assets_dir() / "yolo11n.ckpt", m, strict=True)
    m.eval()
    pred, aux = m(torch.randn(1, 3, 640, 640))

权重转换（.pt → ``assets/ckpts/*.ckpt``）::

    python scripts/convert_ultralytics_weights.py assets/ckpts/yolo11n.pt
    # 或: yolo11-convert-pt assets/ckpts/yolo11n.pt

推理可视化（仅仓库根目录 ``infer.py``）::

    python infer.py assets/ckpts/yolo11n.ckpt path/to/image.jpg
    # 或（pip install -e . 后）: yolo11-infer path/to/image.jpg
"""

from .arch import YOLO11_SCALES
from .layers import Detect
from .model import YOLO11, YOLO11L, YOLO11M, YOLO11N, YOLO11S, YOLO11X
from .paths import assets_dir, ckpt_assets_dir, model_package_dir, project_root
from .utils import guess_scale_from_name, initialize_weights, make_divisible
from .weights import (
    CKPT_FORMAT_VERSION,
    export_ultralytics_pt_to_ckpt,
    load_yolo11_checkpoint,
    save_yolo11_ckpt,
)

__all__ = [
    "YOLO11",
    "YOLO11N",
    "YOLO11S",
    "YOLO11M",
    "YOLO11L",
    "YOLO11X",
    "YOLO11_SCALES",
    "Detect",
    "load_yolo11_checkpoint",
    "save_yolo11_ckpt",
    "export_ultralytics_pt_to_ckpt",
    "CKPT_FORMAT_VERSION",
    "make_divisible",
    "guess_scale_from_name",
    "initialize_weights",
    "assets_dir",
    "ckpt_assets_dir",
    "model_package_dir",
    "project_root",
]
