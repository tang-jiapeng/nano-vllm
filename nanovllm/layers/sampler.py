"""Token 采样器，支持常规采样与 greedy(temperature=0) 路径。"""

import torch
from torch import nn


class Sampler(nn.Module):
    """Token采样器，支持per-sequence温度控制，使用Gumbel-Max技巧实现高效随机采样。"""

    def __init__(self):
        super().__init__()

    def compute_temperature_scaled_probs(
        self, logits: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        """返回温度缩放后的概率分布（float32）。"""
        logits = logits.float()
        safe_temperatures = torch.where(
            temperatures == 0, torch.ones_like(temperatures), temperatures
        )
        logits.div_(safe_temperatures.unsqueeze(dim=1))
        return torch.softmax(logits, dim=-1, dtype=torch.float)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """采样 token；temperature=0 时走 greedy。"""
        probs = self.compute_temperature_scaled_probs(logits, temperatures)
        greedy_tokens = logits.argmax(dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return torch.where(temperatures == 0, greedy_tokens, sampled_tokens)
