"""
YOLO11 检测模型：所有层在 ``__init__`` 里按顺序放进 ``nn.ModuleList``（与官方权重 ``model.0`` … ``model.23`` 对齐），
计算图在 ``forward`` 里逐步写清，无图解析、无 YAML。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .arch import YOLO11_SCALES
from .layers.block import C2PSA, CSPStack2, SPPF
from .layers.conv import Concat, Conv
from .layers.head import Detect
from .utils import initialize_weights, make_divisible

# Detect / neck 用到的固定超参（与 Ultralytics yolo11.yaml 一致）
REG_MAX = 16  # DFL 分布宽度、框分支通道 = 4 * REG_MAX
END2END_DETECT = False
CHANNEL_AXIS = 1  # NCHW 上沿通道维拼接
SPPF_POOL_KERNEL = 5
SPPF_POOL_ITERS = 3  # SPPF 内部串联池化次数（等效多尺度 k=5,9,13）
# 结构里「重复 2 次」的 CSP 块在 YAML 中的名义重复数；再乘 depth_multiple
YAML_REPEAT_TIMES_2 = 2
EXPAND_RATIO_BACKBONE_LO = 0.25  # 前两个 CSP 块的通道展开比例


def _scaled_channels(nominal: int, width_mul: float, max_ch: int) -> int:
    return make_divisible(min(nominal, max_ch) * width_mul, 8)


def _csp_repeat_n(yaml_repeat: int, depth_mul: float) -> int:
    if yaml_repeat > 1:
        return max(round(yaml_repeat * depth_mul), 1)
    return yaml_repeat


def _force_c3k_stack_for_mlx(use_c3k: bool, scale: str) -> bool:
    """n/s 用 YAML 里的布尔；m/l/x 与官方一致，全部走 C3k 子栈。"""
    return True if scale in "mlx" else use_c3k


class YOLO11(nn.Module):
    """
    单尺度 YOLO11。子模块放在 ``self.model``（ModuleList），索引与官方权重一一对应。
    """

    scale: str

    def __init__(self, scale: str, ch: int = 3, nc: int = 80, *, inplace: bool = True):
        super().__init__()
        if scale not in YOLO11_SCALES:
            raise KeyError(scale)
        self.scale = scale
        self.nc = nc
        self.inplace = inplace
        depth_mul, width_mul, max_ch = YOLO11_SCALES[scale]

        sw = lambda n: _scaled_channels(n, width_mul, max_ch)
        n_csp = _csp_repeat_n(YAML_REPEAT_TIMES_2, depth_mul)
        c3k = lambda flag: _force_c3k_stack_for_mlx(flag, scale)

        w64, w128, w256, w512, w1024 = sw(64), sw(128), sw(256), sw(512), sw(1024)

        # neck：每次 Concat 后的输入总通道（可读名字对应 forward 里相接的两路）
        ch_neck_concat_p5up_with_backbone_p4 = w1024 + w512
        ch_neck_concat_p4up_with_backbone_p3 = w512 + w512
        ch_neck_concat_p3down_with_neck_p4 = w256 + w512
        ch_neck_concat_p4down_with_backbone_p5 = w512 + w1024

        # 送入 Detect 的三层特征（P3 / P4 / P5）各自的通道宽
        detect_feat_C_p3, detect_feat_C_p4, detect_feat_C_p5 = w256, w512, w1024

        self.model = nn.ModuleList(
            [
                # --- backbone ---
                Conv(ch, w64, 3, 2),
                Conv(w64, w128, 3, 2),
                CSPStack2(w128, w256, n_csp, c3k(False), EXPAND_RATIO_BACKBONE_LO),
                Conv(w256, w256, 3, 2),
                CSPStack2(w256, w512, n_csp, c3k(False), EXPAND_RATIO_BACKBONE_LO),
                Conv(w512, w512, 3, 2),
                CSPStack2(w512, w512, n_csp, c3k(True)),
                Conv(w512, w1024, 3, 2),
                CSPStack2(w1024, w1024, n_csp, c3k(True)),
                SPPF(w1024, w1024, SPPF_POOL_KERNEL, SPPF_POOL_ITERS),
                C2PSA(w1024, w1024, n_csp),
                # --- head (FPN + PAN) ---
                nn.Upsample(scale_factor=2, mode="nearest"),
                Concat(CHANNEL_AXIS),
                CSPStack2(ch_neck_concat_p5up_with_backbone_p4, w512, n_csp, c3k(False)),
                nn.Upsample(scale_factor=2, mode="nearest"),
                Concat(CHANNEL_AXIS),
                CSPStack2(ch_neck_concat_p4up_with_backbone_p3, w256, n_csp, c3k(False)),
                Conv(w256, w256, 3, 2),
                Concat(CHANNEL_AXIS),
                CSPStack2(ch_neck_concat_p3down_with_neck_p4, w512, n_csp, c3k(False)),
                Conv(w512, w512, 3, 2),
                Concat(CHANNEL_AXIS),
                CSPStack2(ch_neck_concat_p4down_with_backbone_p5, w1024, n_csp, c3k(True)),
                Detect(
                    nc,
                    REG_MAX,
                    END2END_DETECT,
                    (detect_feat_C_p3, detect_feat_C_p4, detect_feat_C_p5),
                ),
            ]
        )

        for i, layer in enumerate(self.model):
            layer.i = i  # type: ignore[attr-defined]
            layer.type = type(layer).__name__  # type: ignore[attr-defined]
        self.model[12].f = [-1, 6]  # type: ignore[attr-defined]
        self.model[15].f = [-1, 4]  # type: ignore[attr-defined]
        self.model[18].f = [-1, 13]  # type: ignore[attr-defined]
        self.model[21].f = [-1, 10]  # type: ignore[attr-defined]
        self.model[23].f = [16, 19, 22]  # type: ignore[attr-defined]
        for i in range(len(self.model)):
            if not hasattr(self.model[i], "f"):
                self.model[i].f = -1  # type: ignore[attr-defined]

        self._init_stride_bias(ch)
        initialize_weights(self)

    def _init_stride_bias(self, ch: int) -> None:
        head = self.model[-1]
        if not isinstance(head, Detect):
            self.stride = torch.tensor([32.0])
            return
        s = 256
        head.inplace = self.inplace
        self.eval()
        head.training = True
        with torch.no_grad():
            out = self.forward(torch.zeros(1, ch, s, s))
            feats = out["feats"]
            strides = torch.tensor([s / t.shape[-2] for t in feats])
        head.stride = strides
        self.stride = strides
        self.train()
        head.bias_init()

    def forward(self, x: torch.Tensor):
        L = self.model

        # --- backbone：得到 P3 / P4 / P5 三档特征（相对输入步长 ×8 / ×16 / ×32）---
        x = L[0](x)
        x = L[1](x)
        x = L[2](x)
        x = L[3](x)
        feat_p3_bb = L[4](x)
        x = L[5](feat_p3_bb)
        feat_p4_bb = L[6](x)
        x = L[7](feat_p4_bb)
        x = L[8](x)
        x = L[9](x)
        feat_p5_bb = L[10](x)

        # --- FPN（自顶向下）：把 P5 语义传到 P4、P3 ---
        x = L[11](feat_p5_bb)
        x = L[12]([x, feat_p4_bb])
        neck_mid_p4 = L[13](x)

        x = L[14](neck_mid_p4)
        x = L[15]([x, feat_p3_bb])
        detect_p3 = L[16](x)

        # --- PAN（再自下而上）：P3→P4→P5，与对应尺度融合 ---
        x = L[17](detect_p3)
        x = L[18]([x, neck_mid_p4])
        detect_p4 = L[19](x)

        x = L[20](detect_p4)
        x = L[21]([x, feat_p5_bb])
        detect_p5 = L[22](x)

        return L[23]([detect_p3, detect_p4, detect_p5])

    @property
    def end2end(self) -> bool:
        h = self.model[-1]
        return getattr(h, "end2end", False) if isinstance(h, Detect) else False


class YOLO11N(YOLO11):
    def __init__(self, ch: int = 3, nc: int = 80, *, inplace: bool = True):
        super().__init__("n", ch=ch, nc=nc, inplace=inplace)


class YOLO11S(YOLO11):
    def __init__(self, ch: int = 3, nc: int = 80, *, inplace: bool = True):
        super().__init__("s", ch=ch, nc=nc, inplace=inplace)


class YOLO11M(YOLO11):
    def __init__(self, ch: int = 3, nc: int = 80, *, inplace: bool = True):
        super().__init__("m", ch=ch, nc=nc, inplace=inplace)


class YOLO11L(YOLO11):
    def __init__(self, ch: int = 3, nc: int = 80, *, inplace: bool = True):
        super().__init__("l", ch=ch, nc=nc, inplace=inplace)


class YOLO11X(YOLO11):
    def __init__(self, ch: int = 3, nc: int = 80, *, inplace: bool = True):
        super().__init__("x", ch=ch, nc=nc, inplace=inplace)
