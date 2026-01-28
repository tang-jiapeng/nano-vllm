"""
RoPE（旋转位置编码）实现

本模块实现了旋转位置编码（Rotary Positional Embedding, RoPE），用于为Transformer模型
添加位置信息。RoPE是一种相对位置编码方法，具有以下优势：

1. 长距离依赖：能处理任意长度的序列（理论上）
2. 相对位置信息：编码的是token之间的相对位置，而非绝对位置
3. 可扩展性：支持线性插值（YaRN等扩展）
4. 高效性：计算简单，内存占用小

核心思想：
将位置信息通过旋转操作应用到Q和K向量上，而不是直接加到嵌入中。
数学公式：对于位置m和维度2i, 2i+1：
- [cos(m*θ_i), -sin(m*θ_i)]
- [sin(m*θ_i),  cos(m*θ_i)]

参考资料：
- RoPE论文：https://arxiv.org/abs/2104.09864
- 原始实现：https://github.com/facebookresearch/llama/blob/main/llama/model.py
"""

from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量

    功能：
    对输入张量x应用RoPE旋转，将位置信息编码到向量中。

    参数：
    - x: 输入张量 [batch_size, num_heads, seq_len, head_dim]
         或 [batch_size * seq_len, num_heads, head_dim]
    - cos: 余弦旋转矩阵 [1, 1, max_seq_len, head_dim]
    - sin: 正弦旋转矩阵 [1, 1, max_seq_len, head_dim]

    返回值：
    - 旋转后的张量，形状同x

    算法：
    1. 将x在最后一个维度上分成两半（偶数索引和奇数索引）
    2. 应用旋转矩阵：
       - y1 = x1 * cos - x2 * sin
       - y2 = x2 * cos + x1 * sin
    3. 合并两半并保持原始数据类型

    数学原理：
    对于二维空间中的向量[x1, x2]，旋转角度θ后的坐标为：
    [x1', x2'] = [x1, x2] @ [[cos θ, -sin θ], [sin θ, cos θ]]

    在RoPE中，θ = m * θ_i，其中m是位置索引，θ_i是基础频率。

    旋转公式：
    ```
    原始向量：[x_even, x_odd]
    旋转后： [x_even * cos - x_odd * sin, x_odd * cos + x_even * sin]
    ```

    使用示例：
    ```python
    # 创建测试数据
    x = torch.randn(2, 4, 8, 64)  # [batch, heads, seq, dim]
    cos = torch.randn(1, 1, 8, 64)
    sin = torch.randn(1, 1, 8, 64)

    # 应用RoPE
    x_rotated = apply_rotary_emb(x, cos, sin)
    ```
    """
    # 第1步：将x在最后一个维度上分成两半
    # 例如：head_dim=64 -> [32, 32]
    # 偶数索引维度（0, 2, 4, ...）和奇数索引维度（1, 3, 5, ...）
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)

    # 第2步：应用旋转矩阵
    # 旋转公式：
    # y1 = x1 * cos - x2 * sin
    # y2 = x2 * cos + x1 * sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    # 第3步：合并两半并转换回原始数据类型
    # 拼接：[y1, y2] -> 原始形状
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）层

    核心功能：
    1. 预计算所有位置的cos和sin值
    2. 高效缓存，避免重复计算
    3. 支持任意序列长度
    4. 与flash-attention等优化兼容

    工作原理：
    - 在初始化时预计算cos(m*θ_i)和sin(m*θ_i)对所有位置m和维度i
    - 在前向传播时，根据positions索引获取对应的cos/sin值
    - 调用apply_rotary_emb将旋转应用到Q和K

    优势：
    1. 内存换时间：预计算所有位置的值，推理时O(1)访问
    2. 并行友好：所有位置的cos/sin可以并行计算
    3. 数值稳定：使用缓存避免重复计算浮点误差

    使用场景：
    - Transformer的注意力层
    - 为Q和K添加位置信息
    - 长序列处理（相比绝对位置编码无长度限制）
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """
        初始化RoPE编码器。

        参数：
        - head_size: 注意力头的维度（如：128）
        - rotary_dim: 旋转的维度数量（通常等于head_size）
        - max_position_embeddings: 支持的最大序列长度（如：131072）
        - base: RoPE的基础频率（通常为10000或1000000）

        预计算流程：
        1. 计算频率：inv_freq = 1 / (base^(2i/dim))
        2. 生成位置：t = [0, 1, 2, ..., max_position-1]
        3. 计算频率矩阵：freqs = outer(t, inv_freq)
        4. 计算cos和sin：cos = cos(freqs), sin = sin(freqs)
        5. 拼接并缓存

        数学公式：
        ```
        # 基础频率（每个维度的角速度）
        inv_freq[i] = 1 / (base ^ (2*i / rotary_dim))

        # 位置频率矩阵
        freqs[m, i] = m * inv_freq[i] = m / (base ^ (2*i / rotary_dim))

        # 最终的cos和sin
        cos[m, i] = cos(freqs[m, i])
        sin[m, i] = sin(freqs[m, i])
        ```

        内存布局：
        - cos_sin_cache: [max_position, 1, head_size * 2]
        - 前head_size个值是cos，后head_size个值是sin

        使用示例：
        ```python
        rope = RotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=131072,
            base=1000000,
        )
        ```
        """
        super().__init__()

        # 验证参数
        self.head_size = head_size
        assert rotary_dim == head_size

        # 第1步：计算基础频率（每个维度的角速度）
        # 公式：inv_freq[i] = 1 / (base ^ (2*i / rotary_dim))
        # 解释：每2个维度共享一个频率（偶数和奇数索引配对）
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))

        # 第2步：生成位置索引
        t = torch.arange(max_position_embeddings, dtype=torch.float)

        # 第3步：计算频率矩阵（位置和维度的外积）
        # freqs[m, i] = t[m] * inv_freq[i]
        # 结果形状：[max_position, rotary_dim/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # 第4步：计算cos和sin
        cos = freqs.cos()  # [max_position, rotary_dim/2]
        sin = freqs.sin()  # [max_position, rotary_dim/2]

        # 第5步：拼接cos和sin
        # 在最后一个维度上拼接：[cos, sin]
        # 形状变为：[max_position, rotary_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)

        # 第6步：注册为buffer（不参与梯度更新，但会保存到state_dict）
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        # 持久化=False意味着：
        # 1. 不会保存到模型权重文件
        # 2. 可以在初始化时重新计算
        # 3. 节省磁盘空间

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：应用RoPE到Q和K。

        参数：
        - positions: 位置索引 [batch_size, seq_len]
        - query: 查询向量 [batch_size * seq_len, num_heads, head_dim]
        - key: 键向量 [batch_size * seq_len, num_kv_heads, head_dim]

        返回值：
        - query: 应用RoPE后的查询向量，形状同输入
        - key: 应用RoPE后的键向量，形状同输入

        工作流程：
        1. 根据positions索引获取对应的cos和sin值
        2. 将cos_sin切分为cos和sin两部分
        3. 分别应用到query和key上
        4. 返回旋转后的向量

        内存布局详解：
        - positions: [batch_size, seq_len]
        - cos_sin_cache: [max_position, 1, head_size * 2]
        - positions索引后: [batch_size, seq_len, head_size * 2]
        - 切分后:
          - cos: [batch_size, seq_len, head_size]
          - sin: [batch_size, seq_len, head_size]
        - 应用到query: query形状需要能广播匹配
          - query: [batch_size * seq_len, num_heads, head_dim]
          - cos/sin: [batch_size, seq_len, head_dim] -> 广播到[batch_size * seq_len, 1, head_dim]

        使用示例：
        ```python
        rope = RotaryEmbedding(...)
        positions = torch.tensor([[0, 1, 2], [0, 1, 2]])  # 2个序列，每个3个位置
        query = torch.randn(6, 8, 128)  # [2*3, 8 heads, 128 dim]
        key = torch.randn(6, 2, 128)    # [2*3, 2 kv heads, 128 dim]
        query_rot, key_rot = rope(positions, query, key)
        ```

        注意事项：
        - positions的值应该小于等于max_position_embeddings
        - query和key的最后一个维度必须等于head_size
        - positions的形状会被广播以匹配query/key的批量大小
        """
        # 第1步：根据位置索引获取cos和sin
        # positions: [batch_size, seq_len]
        # cos_sin_cache: [max_position, 1, head_size * 2]
        # 索引后: [batch_size, seq_len, head_size * 2]
        # 注意：buffer可以使用索引，PyTorch的内部实现支持
        cos_sin = self.cos_sin_cache[positions]  # type: ignore

        # 第2步：切分cos和sin
        # 在最后一个维度上切分
        # cos: [batch_size, seq_len, head_size]
        # sin: [batch_size, seq_len, head_size]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 第3步：应用RoPE到query和key
        # 广播机制：
        # - cos/sin: [batch_size, seq_len, head_dim]
        # - query: [batch_size * seq_len, num_heads, head_dim]
        # 在内存中连续存储，所以可以广播
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)

        # 返回旋转后的向量
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取RoPE编码器实例（带缓存）

    功能：
    使用LRU缓存创建并返回RoPE编码器实例，避免重复创建相同配置的编码器。

    参数：
    - head_size: 注意力头维度
    - rotary_dim: 旋转维度（通常等于head_size）
    - max_position: 最大序列长度
    - base: 基础频率
    - rope_scaling: RoPE缩放配置（当前未实现，默认为None）

    返回值：
    - RotaryEmbedding实例

    LRU缓存机制：
    - 最近最少使用（Least Recently Used）
    - maxsize=1：只缓存最新的1个实例
    - 当参数相同时，直接返回缓存的实例
    - 当参数不新时，创建新实例并替换缓存

    使用场景：
    - 在模型初始化时调用
    - 多个层共享相同的RoPE配置
    - 避免重复分配内存

    示例：
    ```python
    # 第一次调用，创建实例
    rope1 = get_rope(128, 128, 131072, 1000000)

    # 参数相同，使用缓存
    rope2 = get_rope(128, 128, 131072, 1000000)
    assert rope1 is rope2  # 相同的对象

    # 参数不同，创建新实例
    rope3 = get_rope(128, 128, 65536, 1000000)  # max_position不同
    assert rope1 is not rope3
    ```

    注意事项：
    - 当前实现不支持rope_scaling（线性插值等扩展）
    - assert rope_scaling is None会检查此参数
    - 如需支持rope_scaling，需要修改此函数
    """
    # 检查rope_scaling参数（当前不支持）
    assert rope_scaling is None

    # 创建RoPE编码器实例
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)

    # 返回实例（自动缓存）
    return rotary_emb
