"""
性能基准：对比自研 Triton Attention 算子 与 flash-attn 第三方库的吞吐量。

两组 Benchmark：
  1. Prefill  — flash_atten_prefill       vs flash_attn_varlen_func
     X 轴：seq_len（单序列总 token 数），固定 batch=1
     Y 轴：TFLOPS（理论浮点计算量 / 实际耗时）

  2. Decode   — paged_attention_decode    vs flash_attn_with_kvcache
     X 轴：context_len（历史 KV 长度），固定 batch_size=BATCH_SIZE
     Y 轴：TFLOPS
     注：flash_attn 要求 block_size ≥ 256，Triton 使用 block_size=16

参考配置接近 Qwen3-8B（16 heads, 8 KV-heads, 128 dim）。
"""

import math

import torch
import triton
import triton.testing
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.kernels.attention import flash_atten_prefill, paged_attention_decode

# ── 全局超参 ──────────────────────────────────────────────────────────────────
NUM_HEADS = 16  # Q heads
NUM_KV_HEADS = 8  # KV heads（GQA=2）
HEAD_DIM = 128
TRITON_BLOCK_SIZE = 16  # Triton paged KV cache 块大小
FA_BLOCK_SIZE = 256  # flash-attn 要求 block_size 能被 256 整除
BATCH_SIZE = 8  # Decode 阶段的 batch size
DTYPE = torch.float16
DEVICE = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1：Prefill
#   变量：seq_len（单批次所有 token 数，视为一条长序列）
# ─────────────────────────────────────────────────────────────────────────────


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "flash_attn"],
        line_names=["Triton", "Flash-Attn"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="prefill-attention-performance",
        args={
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
        },
    )
)
def bench_prefill(seq_len, num_heads, num_kv_heads, head_dim, provider):
    """Prefill：单序列因果注意力吞吐量对比。"""
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(seq_len, num_heads, head_dim, dtype=DTYPE, device=DEVICE)
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_atten_prefill(
                q, k, v, cu, scale, num_heads, num_kv_heads, head_dim
            ),
            quantiles=quantiles,
        )
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                softmax_scale=scale,
                causal=True,
            ),
            quantiles=quantiles,
        )

    # ── TFLOPS 计算 ──────────────────────────────────────────────────────────
    # 因果注意力：有效参与计算的 QK 对约为 T*(T+1)/2 ≈ T^2/2
    # QK^T + PV 两次矩阵乘法，每次 2 FLOPs/element
    #   → 总计 ≈ 2 × (T^2/2) × num_heads × head_dim × 2 = 2T^2 × nh × hd
    flops = 2 * (seq_len**2) * num_heads * head_dim
    tflops = lambda ms: flops * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2：Decode（Paged KV Cache）
#   变量：context_len（历史 KV 长度，当前 token 对其做 attention）
# ─────────────────────────────────────────────────────────────────────────────


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["context_len"],
        x_vals=[64, 128, 256, 512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "flash_attn"],
        line_names=["Triton", "Flash-Attn"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="decode-attention-performance",
        args={
            "batch_size": BATCH_SIZE,
            "num_heads": NUM_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
        },
    )
)
def bench_decode(context_len, batch_size, num_heads, num_kv_heads, head_dim, provider):
    """Decode：paged attention 对 context_len 历史 KV 的吞吐量对比。"""
    scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(batch_size, num_heads, head_dim, dtype=DTYPE, device=DEVICE)
    ctx_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=DEVICE)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "triton":
        # Triton: block_size=16（原生设计）
        bs = TRITON_BLOCK_SIZE
        blocks_per_seq = math.ceil(context_len / bs)
        total_blocks = batch_size * blocks_per_seq
        block_tables = torch.arange(
            total_blocks, device=DEVICE, dtype=torch.int32
        ).reshape(batch_size, blocks_per_seq)
        k_cache = torch.randn(
            total_blocks, bs, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE
        )
        v_cache = torch.randn(
            total_blocks, bs, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE
        )

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: paged_attention_decode(
                q, k_cache, v_cache, block_tables, ctx_lens, scale
            ),
            quantiles=quantiles,
        )
    else:
        # Flash-Attn: block_size=256（库要求 ≥256 且能被 256 整除）
        bs = FA_BLOCK_SIZE
        blocks_per_seq = math.ceil(context_len / bs)
        total_blocks = batch_size * blocks_per_seq
        block_tables = torch.arange(
            total_blocks, device=DEVICE, dtype=torch.int32
        ).reshape(batch_size, blocks_per_seq)
        k_cache = torch.randn(
            total_blocks, bs, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE
        )
        v_cache = torch.randn(
            total_blocks, bs, num_kv_heads, head_dim, dtype=DTYPE, device=DEVICE
        )

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=ctx_lens,
                block_table=block_tables,
                softmax_scale=scale,
                causal=True,
            ),
            quantiles=quantiles,
        )

    # ── TFLOPS 计算 ──────────────────────────────────────────────────────────
    # Decode：每个 batch 中，单个新 token 对 context_len 个历史 token 做 attention
    #   QK^T: batch × context_len × num_heads × head_dim × 2 FLOPs
    #   PV  : 同上
    #   合计: 4 × batch × context_len × num_heads × head_dim
    flops = 4 * batch_size * context_len * num_heads * head_dim
    tflops = lambda ms: flops * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(">>> Prefill Attention Benchmark")
    bench_prefill.run(print_data=True, save_path=".")

    print("\n>>> Decode Attention Benchmark")
    bench_decode.run(print_data=True, save_path=".")
