import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU激活函数，计算 silu(x) * y，其中 x, y 为输入张量的两半。"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输入最后一维为 2*intermediate_size，输出 SwiGLU 激活结果。"""
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
