"""RMSNorm 实现，现代大语言模型的标准归一化方法，相比 LayerNorm 省去均值中心化步骤，计算更高效。"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm 层归一化，支持标准前向和带残差连接的前向传播，使用 torch.compile 加速。"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """初始化 RMSNorm 层。"""
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """标准 RMSNorm 前向传播，在 float32 精度下计算归一化后转回原始 dtype。"""
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """带残差连接的 RMSNorm 前向传播，原地将 x 与 residual 相加后归一化。"""
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)

        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """统一前向接口，根据 residual 是否为 None 选择标准或带残差的归一化。"""
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
