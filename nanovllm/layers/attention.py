"""
Flash Attention实现 - 高效的多头注意力计算

本模块实现了优化版的Transformer注意力机制，集成了：
1. Flash Attention - 减少内存占用和计算复杂度
2. KV-Cache - 支持增量推理
3. Prefix Caching - 复用历史计算结果
4. Triton内核 - 高效的KV-cache存储

核心优势：
- 内存效率：Flash Attention将内存复杂度从O(N²)降至O(N)
- 计算效率：使用Triton内核优化KV-cache写入
- 推理优化：支持prefill和decode两种模式自动切换

参考资料：
- Flash Attention论文：https://arxiv.org/abs/2205.14135
- Flash Attention 2：https://arxiv.org/abs/2307.08691
- Prefix Caching：vLLM的实现
"""

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton JIT内核：将K和V存储到KV-cache

    功能：
    - 批量并行地将当前计算的K和V写入KV-cache
    - 支持非连续内存访问
    - 使用Triton优化内存读写性能

    参数说明：
    - key_ptr/value_ptr: 当前K、V的指针
    - key_stride/value_stride: 当前K、V的步长
    - k_cache_ptr/v_cache_ptr: KV-cache的指针
    - slot_mapping_ptr: 槽位映射指针，指示每个token应该写入cache的哪个位置
    - D: 维度大小 (num_heads * head_dim)

    工作原理：
    1. 获取当前线程块ID（对应一个token）
    2. 从slot_mapping加载该token应该写入的cache槽位
    3. 如果slot为-1（无效），直接返回
    4. 计算K、V和cache的偏移量
    5. 加载K、V值并写入对应的cache位置

    内存布局：
    - K、V: [N, num_heads, head_dim]，要求最后两个维度连续存储
    - KV-cache: [num_blocks, block_size, num_heads, head_dim]
    - slot_mapping: [N]，每个token对应的cache槽位

    注意：
    - 使用tl.constexpr声明D为编译时常量
    - 槽位为-1时跳过写入（无效token）
    """
    # 获取当前线程块ID（对应一个token的索引）
    idx = tl.program_id(0)

    # 加载该token对应的cache槽位
    slot = tl.load(slot_mapping_ptr + idx)

    # 槽位为-1表示无效token，直接返回
    if slot == -1:
        return

    # 计算K和V的偏移量
    # 每个token的起始位置 = idx * key_stride
    # 当前维度偏移 = tl.arange(0, D)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # 加载K和V的值
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 计算cache的偏移量
    # cache位置 = slot * D + tl.arange(0, D)
    cache_offsets = slot * D + tl.arange(0, D)

    # 将K和V写入cache
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    将K和V存储到KV-cache（Python包装函数）

    使用方法：
    ```python
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)
    ```

    参数：
    - key: 当前的K tensor [N, num_heads, head_dim]
    - value: 当前的V tensor [N, num_heads, head_dim]
    - k_cache: K-cache [num_blocks, block_size, num_heads, head_dim]
    - v_cache: V-cache [num_blocks, block_size, num_heads, head_dim]
    - slot_mapping: 槽位映射 [N]，每个元素表示对应token应写入的cache槽位索引

    功能说明：
    - 验证输入张量的内存布局符合要求
    - 调用Triton内核并行写入KV-cache
    - 支持非连续内存布局（通过步长计算偏移）

    内存布局要求：
    1. K和V的最后一个维度必须连续（stride[-1] == 1）
    2. K和V的第二个维度步长必须等于head_dim（行连续）
    3. KV-cache的第二个维度步长必须等于num_heads * head_dim
    4. slot_mapping长度必须等于N（token数量）

    性能优化：
    - 使用Triton JIT编译加速
    - 并行写入多个token
    - 原地操作，无额外内存分配

    示例：
    ```python
    # 假设有4个token，每个token有8个头，每头64维
    N, num_heads, head_dim = 4, 8, 64
    k = torch.randn(N, num_heads, head_dim)
    v = torch.randn(N, num_heads, head_dim)

    # KV-cache有100个块，每块16个token
    num_blocks, block_size = 100, 16
    k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
    v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)

    # 每个token对应的槽位
    slot_mapping = torch.tensor([0, 0, 1, 1])  # 前2个token在块0，后2个在块1

    # 写入cache
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)
    ```

    注意事项：
    - 确保KV-cache有足够的空间
    - slot_mapping中的值必须在[0, num_blocks)范围内
    - K和V的数值类型应该与cache一致（通常是float16或bfloat16）
    """
    # 验证输入张量形状
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 验证内存布局（确保连续性）
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D

    # 验证slot_mapping长度
    assert slot_mapping.numel() == N

    # 调用Triton内核
    # [(N,)]表示启动N个线程块，每个线程块处理一个token
    store_kvcache_kernel[(N,)](
        key,
        key.stride(0),  # 当前K及其步长
        value,
        value.stride(0),  # 当前V及其步长
        k_cache,
        v_cache,  # KV-cache
        slot_mapping,  # 槽位映射
        D,  # 维度大小（编译时常量）
    )


class Attention(nn.Module):
    """
    Flash Attention层 - 支持KV-cache和prefix caching

    核心功能：
    1. 多头注意力计算（使用Flash Attention优化）
    2. KV-cache自动存储和管理
    3. Prefix caching支持（复用历史计算）
    4. 自动模式选择（prefill vs decode）

    两种推理模式：
    1. Prefill模式（填充模式）：
       - 处理输入prompt的所有tokens
       - 使用flash_attn_varlen_func，支持变长序列
       - 支持prefix caching（检测并复用相同前缀）
       - 计算所有位置的注意力

    2. Decode模式（解码模式）：
       - 生成新的token（一次只生成1个）
       - 使用flash_attn_with_kvcache，利用历史KV
       - 只计算最后一个位置的注意力
       - 高效的增量计算

    性能优化：
    - Flash Attention：内存复杂度O(1) vs 标准attention O(N)
    - KV-cache：避免重复计算历史tokens的K和V
    - Prefix caching：检测相同前缀，复用计算结果
    - Triton内核：优化KV-cache写入速度

    使用场景：
    - Transformer解码器的注意力层
    - 大语言模型推理（prefill + decode）
    - 批量请求处理
    - 长序列处理（支持超长上下文）
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        """
        初始化注意力层。

        参数：
        - num_heads: Q的注意力头数量
        - head_dim: 每个头的维度
        - scale: 注意力缩放因子（通常为head_dim的-0.5次方）
        - num_kv_heads: K和V的注意力头数量（用于GQA）

        注意：
        - KV-cache在初始化时为空，运行时由外部注入
        - k_cache和v_cache是buffer，不是parameter（不参与训练）
        """
        super().__init__()

        # 注意力头配置
        self.num_heads = num_heads  # Q的头数
        self.head_dim = head_dim  # 每头的维度
        self.scale = scale  # 缩放因子（防止softmax饱和）
        self.num_kv_heads = num_kv_heads  # K、V的头数（GQA）

        # 初始化空的KV-cache
        # 这些buffer会在运行时被赋值
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：计算注意力输出

        参数：
        - q: 查询向量 [N, num_heads, head_dim]
        - k: 键向量 [N, num_kv_heads, head_dim]
        - v: 值向量 [N, num_kv_heads, head_dim]

        返回值：
        - o: 注意力输出 [N, num_heads, head_dim]

        工作流程：
        1. 获取推理上下文（包含cache、序列信息等）
        2. 如果KV-cache存在，将当前K、V写入cache
        3. 根据模式选择（prefill/decode）调用不同的attention kernel
        4. 返回注意力输出

        上下文信息：
        - is_prefill: 是否为prefill模式
        - slot_mapping: token到cache槽位的映射
        - block_tables: prefix caching的块表
        - max_seqlen_q/k: query和key的最大序列长度
        - cu_seqlens_q/k: cumulative sequence lengths
        - context_lens: 每个序列的上下文长度

        Flash Attention参数：
        - causal=True: 使用因果注意力（不能看未来）
        - softmax_scale: 缩放因子，防止softmax饱和
        - block_table: prefix caching的块表
        """
        # 获取推理上下文
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 步骤1：如果KV-cache存在，将当前K、V写入cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 步骤2：根据模式选择不同的attention kernel
        if context.is_prefill:
            # ========== Prefill模式 ==========
            # 处理输入prompt的所有tokens

            # Prefix caching：如果block_tables存在，说明支持prefix caching
            if context.block_tables is not None:
                # 使用已缓存的K、V（包含历史计算结果）
                k, v = k_cache, v_cache

            # 调用flash-attention varlen kernel
            # 支持变长序列和prefix caching
            o = flash_attn_varlen_func(
                q,
                k,
                v,  # Q、K、V
                max_seqlen_q=context.max_seqlen_q,  # query的最大序列长度
                cu_seqlens_q=context.cu_seqlens_q,  # query的累计长度
                max_seqlen_k=context.max_seqlen_k,  # key的最大序列长度
                cu_seqlens_k=context.cu_seqlens_k,  # key的累计长度
                softmax_scale=self.scale,  # 缩放因子
                causal=True,  # 因果注意力
                block_table=context.block_tables,  # prefix caching块表
            )
        else:
            # ========== Decode模式 ==========
            # 生成新token（一次一个）

            # 在query维度上添加一个维度 [N, num_heads, head_dim] -> [N, 1, num_heads, head_dim]
            # flash_attn_with_kvcache期望这种格式
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),  # 增加一个维度
                k_cache,  # 使用缓存的K
                v_cache,  # 使用缓存的V
                cache_seqlens=context.context_lens,  # 每个序列的上下文长度
                block_table=context.block_tables,  # block table
                softmax_scale=self.scale,  # 缩放因子
                causal=True,  # 因果注意力
            )

        # 返回注意力输出
        return o
