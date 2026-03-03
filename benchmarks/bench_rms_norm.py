import torch
import triton
import triton.testing
from torch import nn
from nanovllm.kernels.rms_norm import rms_norm_fused

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)

        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            # bench 中主要测 fused，此处略去 rms_forward 具体实现
            pass
        else:
            return self.add_rms_forward(x, residual)

# ---------------------------------------------------------------------------
# Benchmark 核心逻辑
# ---------------------------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_tokens'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['torch_compile', 'triton'],
        line_names=['PyTorch (@torch.compile)', 'Triton (Hand-written)'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='rms-norm-performance',
        args={'hidden_size': 4096}
    )
)
def benchmark(num_tokens, hidden_size, provider):
    # 初始化数据
    x = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float16)
    residual = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float16)
    eps = 1e-6

    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch_compile':
        # 实例化实际的 RMSNorm 模型并放到 GPU
        rms_module = RMSNorm(hidden_size, eps=eps).cuda().to(torch.float16)
        
        # 预热 torch.compile
        _ = rms_module(x, residual)

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms_module(x, residual), quantiles=quantiles
        )
        
    elif provider == 'triton':
        # Triton kernel 需要提取 weight tensor
        weight = torch.ones(hidden_size, device='cuda', dtype=torch.float16)
        
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms_norm_fused(x, residual, weight, eps), quantiles=quantiles
        )

    # -------------------------------------------------------------
    # 计算理论显存访问量 (Bytes)
    # 读: x (2 bytes), residual (2 bytes)
    # 写: y (2 bytes), r_out (2 bytes)
    # -------------------------------------------------------------
    gbps = lambda ms: (4 * x.numel() * x.element_size()) / (ms * 1e-3) / 1e9
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == '__main__':
    print("开始运行 Benchmark ...")
    benchmark.run(print_data=True, save_path='.')