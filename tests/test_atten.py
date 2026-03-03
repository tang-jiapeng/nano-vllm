"""
正确性验证：对比自研 Triton Attention 算子 与 flash-attn 第三方库的输出。

覆盖场景：
  1. Prefill  — flash_atten_prefill       vs flash_attn_varlen_func
  2. Decode   — paged_attention_decode    vs flash_attn_with_kvcache

两个场景均覆盖：MHA / GQA、单序列 / 多序列、不同 head_dim 和 block_size。
"""

import math

import torch
from flash_attn import flash_attn_varlen_func

from nanovllm.kernels.attention import flash_atten_prefill, paged_attention_decode

# ── 纯 PyTorch 参考（无第三方依赖，用于双重核验）─────────────────────────────


def _ref_prefill_torch(q, k, v, cu_seqlens, scale):
    """逐序列做因果 Self-Attention 的 PyTorch 参考实现（支持 GQA）。"""
    results = []
    for i in range(len(cu_seqlens) - 1):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        qi = q[s:e]  # [L, nh,  hd]
        ki = k[s:e]  # [L, nkh, hd]
        vi = v[s:e]
        L = e - s

        # GQA：将 KV head 重复扩展到与 Q 一致
        gqa = qi.shape[1] // ki.shape[1]
        if gqa > 1:
            ki = ki.repeat_interleave(gqa, dim=1)
            vi = vi.repeat_interleave(gqa, dim=1)

        qi = qi.permute(1, 0, 2).float()  # [nh, L, hd]
        ki = ki.permute(1, 0, 2).float()
        vi = vi.permute(1, 0, 2).float()

        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale  # [nh, L, L]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, vi).permute(1, 0, 2).to(q.dtype)  # [L, nh, hd]
        results.append(out)

    return torch.cat(results, dim=0)


def _ref_decode_torch(q, k_cache, v_cache, block_tables, context_lens, scale):
    """
    逐 batch 做 PagedAttention Decode 的 PyTorch 参考实现（支持 GQA）。
    读取物理 KV cache，按 block_tables 还原逻辑 KV，再做 dot-product attention。
    """
    batch, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    gqa = num_heads // num_kv_heads

    results = []
    for b in range(batch):
        ctx_len = int(context_lens[b])
        q_b = q[b].float()  # [nh, hd]

        # 按页表还原完整 KV
        blocks_needed = math.ceil(ctx_len / block_size)
        k_list, v_list = [], []
        for blk_idx in range(blocks_needed):
            phys = int(block_tables[b, blk_idx])
            k_list.append(k_cache[phys])  # [bs, nkh, hd]
            v_list.append(v_cache[phys])

        k_all = torch.cat(k_list, dim=0)[:ctx_len].float()  # [ctx, nkh, hd]
        v_all = torch.cat(v_list, dim=0)[:ctx_len].float()

        # GQA 扩展
        if gqa > 1:
            k_all = k_all.repeat_interleave(gqa, dim=1)
            v_all = v_all.repeat_interleave(gqa, dim=1)

        k_all = k_all.permute(1, 0, 2)  # [nh, ctx, hd]
        v_all = v_all.permute(1, 0, 2)

        # q_b: [nh, hd]  →  unsqueeze → [nh, 1, hd]
        scores = (
            torch.matmul(q_b.unsqueeze(1), k_all.transpose(-2, -1)) * scale
        )  # [nh, 1, ctx]
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_all).squeeze(1).to(q.dtype)  # [nh, hd]
        results.append(out)

    return torch.stack(results, dim=0)  # [B, nh, hd]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Prefill 正确性测试
# ─────────────────────────────────────────────────────────────────────────────

PREFILL_CASES = [
    # (seq_lens,   num_heads, num_kv_heads, head_dim, 描述)
    ([8], 8, 8, 64, "MHA 单序列"),
    ([4, 7, 5], 8, 8, 64, "MHA 三序列 batch"),
    ([8], 8, 2, 64, "GQA=4 单序列"),
    ([4, 7], 8, 2, 64, "GQA=4 双序列"),
    ([16], 16, 4, 128, "GQA=4 大 head_dim=128"),
    ([32], 8, 8, 128, "MHA seq=32 head_dim=128"),
]


def test_prefill_correctness():
    """比较 flash_atten_prefill (Triton) 与 flash_attn_varlen_func (参考)。"""
    print("\n" + "=" * 60)
    print("Prefill Attention 正确性验证")
    print("=" * 60)

    passed = failed = 0
    for seq_lens, num_heads, num_kv_heads, head_dim, desc in PREFILL_CASES:
        tri = ref_fa = None
        try:
            torch.manual_seed(42)
            device, dtype = "cuda", torch.float16
            scale = 1.0 / math.sqrt(head_dim)
            total = sum(seq_lens)

            cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
            for i, l in enumerate(seq_lens):
                cu[i + 1] = cu[i] + l

            q = torch.randn(total, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(total, num_kv_heads, head_dim, dtype=dtype, device=device)

            # ── 参考 1：flash_attn_varlen_func ──────────────────────────────
            ref_fa = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu,
                cu_seqlens_k=cu,
                max_seqlen_q=max(seq_lens),
                max_seqlen_k=max(seq_lens),
                softmax_scale=scale,
                causal=True,
            )

            # ── 参考 2：纯 PyTorch（双重核验）──────────────────────────────
            ref_pt = _ref_prefill_torch(q, k, v, cu, scale)

            # ── 待测：自研 Triton 算子 ──────────────────────────────────────
            tri = flash_atten_prefill(
                q, k, v, cu, scale, num_heads, num_kv_heads, head_dim
            )

            # flash_attn 与 PyTorch 参考应高度一致
            torch.testing.assert_close(ref_fa, ref_pt, atol=1e-2, rtol=1e-2)
            # Triton 与 flash_attn 参考对比
            torch.testing.assert_close(tri, ref_fa, atol=1e-2, rtol=1e-2)
            print(f"  ✅  {desc}")
            passed += 1
        except AssertionError as e:
            max_err = (
                (tri - ref_fa).abs().max().item()
                if (tri is not None and ref_fa is not None)
                else float("nan")
            )
            print(f"  ❌  {desc}  （结果不一致，最大绝对误差 = {max_err:.5f}）")
            print(f"      {str(e)[:300]}")
            failed += 1
        except Exception as e:
            print(f"  ❌  {desc}  （运行时异常：{type(e).__name__}: {str(e)[:200]}）")
            failed += 1

    print(f"\n[Prefill] 通过 {passed}/{passed+failed} 个用例\n")
    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Decode 正确性测试
# ─────────────────────────────────────────────────────────────────────────────

DECODE_CASES = [
    # (batch, ctx_len, num_heads, num_kv_heads, head_dim, block_size, 描述)
    (1, 16, 8, 8, 64, 16, "MHA batch=1  ctx=16"),
    (4, 32, 8, 8, 64, 16, "MHA batch=4  ctx=32"),
    (2, 48, 8, 8, 64, 16, "MHA batch=2  ctx=48（非 block 对齐）"),
    (4, 64, 8, 8, 64, 16, "MHA batch=4  ctx=64"),
    (4, 32, 8, 2, 64, 16, "GQA=4 batch=4"),
    (4, 64, 16, 4, 128, 16, "GQA=4 head_dim=128"),
]


def test_decode_correctness():
    """比较 paged_attention_decode (Triton) 与 flash_attn_with_kvcache (参考)。"""
    print("=" * 60)
    print("Decode Paged Attention 正确性验证")
    print("=" * 60)

    passed = failed = 0
    for (
        batch,
        ctx_len,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        desc,
    ) in DECODE_CASES:
        tri = ref_fa = None
        try:
            torch.manual_seed(0)
            device, dtype = "cuda", torch.float16
            scale = 1.0 / math.sqrt(head_dim)

            blocks_per_seq = math.ceil(ctx_len / block_size)
            total_blocks = batch * blocks_per_seq
            block_tables = torch.arange(
                total_blocks, device=device, dtype=torch.int32
            ).reshape(batch, blocks_per_seq)
            k_cache = torch.randn(
                total_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            v_cache = torch.randn(
                total_blocks,
                block_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device)
            ctx_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device=device)

            # ── 参考：纯 PyTorch 实现 ──────────────────────────────────────
            ref_pt = _ref_decode_torch(
                q, k_cache, v_cache, block_tables, ctx_lens, scale
            )

            # ── 待测：自研 Triton 算子 ──────────────────────────────────────
            tri = paged_attention_decode(
                q, k_cache, v_cache, block_tables, ctx_lens, scale
            )

            torch.testing.assert_close(tri, ref_pt, atol=1e-2, rtol=1e-2)
            print(f"  ✅  {desc}")
            passed += 1
        except AssertionError as e:
            max_err = (
                (tri - ref_pt).abs().max().item()
                if (tri is not None and ref_pt is not None)
                else float("nan")
            )
            print(f"  ❌  {desc}  （结果不一致，最大绝对误差 = {max_err:.5f}）")
            print(f"      {str(e)[:300]}")
            failed += 1
        except Exception as e:
            print(f"  ❌  {desc}  （运行时异常：{type(e).__name__}: {str(e)[:200]}）")
            failed += 1

    print(f"\n[Decode] 通过 {passed}/{passed+failed} 个用例\n")
    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok_prefill = test_prefill_correctness()
    ok_decode = test_decode_correctness()
    if ok_prefill and ok_decode:
        print("🎉 所有用例均通过！")
    else:
        print("⚠️  存在失败用例，请检查 Triton 算子实现。")
