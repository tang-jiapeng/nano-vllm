from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.kernels.awq_gemm import awq_gemm_triton

if TYPE_CHECKING:
    from nanovllm.layers.linear import LinearBase


class LinearMethodBase(ABC):
    """线性层量化策略基类，定义创建权重和前向计算的接口。"""

    @abstractmethod
    def create_weights(
        self, layer: "LinearBase", input_size: int, output_size: int
    ) -> None:
        """在 layer 上创建权重参数并绑定 weight_loader。"""
        ...

    @abstractmethod
    def apply(
        self,
        layer: "LinearBase",
        x: torch.Tensor,
        bias: Optional[nn.Parameter],
    ) -> torch.Tensor:
        """执行线性前向计算。"""
        ...


class UnquantizedLinearMethod(LinearMethodBase):
    """普通 fp16/bf16 线性层策略。"""

    def create_weights(
        self, layer: "LinearBase", input_size: int, output_size: int
    ) -> None:
        layer.weight = nn.Parameter(torch.empty(output_size, input_size))
        layer.weight.weight_loader = layer.weight_loader

    def apply(
        self,
        layer: "LinearBase",
        x: torch.Tensor,
        bias: Optional[nn.Parameter],
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


@dataclass
class AWQConfig:
    """AWQ 量化配置。"""

    weight_bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    version: str = "GEMM"

    def __post_init__(self):
        if self.weight_bits != 4:
            raise ValueError(
                f"Only AWQ int4 is supported, got weight_bits={self.weight_bits}"
            )
        self.pack_factor = 32 // self.weight_bits  # 8

    @classmethod
    def from_json(cls, model_path: str) -> Optional["AWQConfig"]:
        """
        扫描模型目录，检测 AWQ 量化配置，返回 AWQConfig 或 None。

        检测顺序：
        1. 独立量化配置文件：quant_config.json / quantize_config.json
        2. 嵌入在 config.json 的 quantization_config 字段（HuggingFace 新格式）
        """
        candidate_files = ["quant_config.json", "quantize_config.json"]
        cfg = None
        for filename in candidate_files:
            path = os.path.join(model_path, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                break

        # 回退：检查 config.json 中嵌入的 quantization_config（HuggingFace 格式）
        if cfg is None:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    hf_cfg = json.load(f)
                cfg = hf_cfg.get("quantization_config")

        if cfg is None:
            return None

        method = str(cfg.get("quant_method", cfg.get("quant_type", ""))).lower()
        if "awq" not in method:
            return None

        return cls(
            weight_bits=int(cfg.get("w_bit", cfg.get("bits", 4))),
            group_size=int(cfg.get("q_group_size", cfg.get("group_size", 128))),
            zero_point=bool(cfg.get("zero_point", True)),
            version=str(cfg.get("version", "GEMM")),
        )


class AWQLinearMethod(LinearMethodBase):
    """AWQ 量化策略：动态反量化 + GEMM。"""

    def __init__(self, config: AWQConfig):
        self.config = config

    def create_weights(
        self, layer: "LinearBase", input_size: int, output_size: int
    ) -> None:
        """
        在 layer 上创建 AWQ 量化权重并绑定 weight_loader。
        qweight: [K, N//pack_factor] int32
        qzeros:  [K//group_size, N//pack_factor] int32
        scales:  [K//group_size, N] float16
        """
        g = self.config.group_size
        p = self.config.pack_factor
        if input_size % g != 0:
            raise ValueError(
                f"input_size={input_size} must be divisible by group_size={g}"
            )
        if output_size % p != 0:
            raise ValueError(
                f"output_size={output_size} must be divisible by pack_factor={p}"
            )

        layer.qweight = nn.Parameter(
            torch.empty(input_size, output_size // p, dtype=torch.int32),
            requires_grad=False,
        )
        layer.qzeros = nn.Parameter(
            torch.empty(input_size // g, output_size // p, dtype=torch.int32),
            requires_grad=False,
        )
        layer.scales = nn.Parameter(
            torch.empty(input_size // g, output_size, dtype=torch.float16),
            requires_grad=False,
        )
        for name in ("qweight", "qzeros", "scales"):
            getattr(layer, name).weight_loader = layer.weight_loader

    def _pick_split_k(self, m: int) -> int:
        if m <= 8:
            return 8
        if m <= 32:
            return 4
        return 1

    def apply(
        self,
        layer: "LinearBase",
        x: torch.Tensor,
        bias: Optional[nn.Parameter],
    ) -> torch.Tensor:
        qweight, qzeros, scales = layer.qweight, layer.qzeros, layer.scales
        out_features = qweight.shape[1] * self.config.pack_factor
        out_shape = x.shape[:-1] + (out_features,)
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, x.shape[-1])  # [M, K]
        # Triton AWQ kernel 仅支持 float16，转换输入；输出结束后恢复原始 dtype
        if x_2d.dtype != torch.float16:
            x_2d = x_2d.to(torch.float16)
        split_k = self._pick_split_k(x_2d.shape[0])
        out = awq_gemm_triton(x_2d, qweight, scales, qzeros, split_k_iters=split_k)
        if bias is not None:
            out = out + bias.to(torch.float16)
        return out.reshape(out_shape).to(orig_dtype)
