import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["stride_x"],
)
@triton.jit
def rms_norm_fwd_kernel(
    x_ptr,  # 输入 [num_tokens, hidden_size]
    r_ptr,  # residual [num_tokens, hidden_size]，可为 None（传 0 指针）
    w_ptr,  # 可学习参数 gamma [hidden_size]
    y_ptr,  # 输出 [num_tokens, hidden_size]
    r_out_ptr,  # residual 输出（Y 加 R 后，供下一层用）
    eps: tl.constexpr,
    stride_x: tl.constexpr,  # hidden_size
    BLOCK_SIZE: tl.constexpr,
):
    """
    每个 program 处理一个 token（一行）
    计算：R_out = X + R，Y = R_out / RMS(R_out) * W
    """
    row_idx = tl.program_id(0)

    # 指针偏移到当前行
    x_ptr = x_ptr + row_idx * stride_x
    y_ptr = y_ptr + row_idx * stride_x

    # 加载数据（分 tile 处理大 hidden_size）
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < stride_x

    x = tl.load(x_ptr + cols, mask=mask).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)

    # 计算 fused residual + RMSNorm
    if r_ptr is not None:
        r_ptr = r_ptr + row_idx * stride_x
        r_out_ptr = r_out_ptr + row_idx * stride_x
        r = tl.load(r_ptr + cols, mask=mask).to(tl.float32)
        x_plus_r = x + r
        tl.store(r_out_ptr + cols, x_plus_r.to(x.dtype), mask=mask)
    else:
        x_plus_r = x

    # 计算 RMSNorm
    mean_sq = tl.sum(x_plus_r * x_plus_r, axis=0) / stride_x
    rrms = 1.0 / tl.sqrt(mean_sq + eps)
    y = x_plus_r * rrms * w

    tl.store(y_ptr + cols, y.to(x.dtype), mask=mask)


def rms_norm_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (normed_output, new_residual)"""
    assert x.is_contiguous()
    if residual is not None:
        assert residual.is_contiguous()
        assert x.shape == residual.shape

    num_tokens, hidden_size = x.shape

    y = torch.empty_like(x)
    r_out = torch.empty_like(x) if residual is not None else None

    # BLOCK_SIZE 选择最接近且大于等于 hidden_size 的 2 的幂次
    BLOCK_SIZE = triton.next_power_of_2(hidden_size)

    rms_norm_fwd_kernel[(num_tokens,)](
        x,
        residual,
        weight,
        y,
        r_out,
        eps=eps,
        stride_x=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y, r_out
