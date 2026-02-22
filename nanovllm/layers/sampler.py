"""基于Gumbel-Max技巧的token采样器，通过温度缩放控制分布锐度后从logits中采样token。"""

import torch
from torch import nn


class Sampler(nn.Module):
    """Token采样器，支持per-sequence温度控制，使用Gumbel-Max技巧实现高效随机采样。"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """对logits进行温度缩放后，通过Gumbel-Max技巧采样token。"""
        # 温度缩放
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        probs = torch.softmax(logits, dim=-1)

        # Gumbel-Max采样：probs / exponential(1) 等价于添加Gumbel噪声
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        return sample_tokens
