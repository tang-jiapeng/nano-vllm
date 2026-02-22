"""RoPE（Rotary Positional Embedding）实现，通过旋转操作将位置信息编码到 Q/K 向量中。"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """对输入张量应用 RoPE 旋转，将 x 按最后一维分半后执行旋转变换。"""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE 层，预计算并缓存所有位置的 cos/sin 值，前向传播时按 position 索引取值并应用旋转。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """预计算 inv_freq 并缓存所有位置的 cos/sin 值。"""
        super().__init__()

        self.head_size = head_size
        assert rotary_dim == head_size

        # inv_freq[i] = 1 / (base ^ (2i / dim))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )

        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # freqs[m, i] = t[m] * inv_freq[i]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        cos = freqs.cos()
        sin = freqs.sin()

        # 拼接 [cos, sin] 并缓存为 buffer
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据 positions 索引取 cos/sin，分别应用 RoPE 到 query 和 key。"""
        cos_sin = self.cos_sin_cache[positions]  # type: ignore
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """通过 LRU 缓存获取 RotaryEmbedding 实例，相同配置复用同一对象。"""
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
