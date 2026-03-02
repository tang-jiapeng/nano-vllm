import torch
import triton
import triton.testing
from nanovllm.kernels.rms_norm import rms_norm_fused

def rms_norm_pytorch(x, residual, weight, eps=1e-6):
    """PyTorch 的原生参考实现"""
    x_fp32 = x.float()
    r_fp32 = residual.float()
    x_plus_r = x_fp32 + r_fp32
    rms = torch.sqrt((x_plus_r ** 2).mean(dim=-1, keepdim=True) + eps)
    y = (x_plus_r / rms * weight.float()).to(x.dtype)
    r_out = x_plus_r.to(x.dtype)
    return y, r_out

# 使用 Triton 自带的 Benchmark 工具绘制对比图表
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['num_tokens'],               # X 轴：要测试的变量名
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],  # X 轴的值（代表序列长度或批处理总 token 数）
        line_arg='provider',                  # 用不同的线表示不同的实现
        line_vals=['torch', 'triton'],        # provider 变量的可选值
        line_names=['PyTorch', 'Triton (Fused)'], # 图例名称
        styles=[('blue', '-'), ('green', '-')],   # 颜色和线型
        ylabel='GB/s',                        # Y 轴：显存带宽
        plot_name='rms-norm-performance',     # 绘图保存的文件名前缀
        args={'hidden_size': 4096}            # 固定的参数：模拟常见大模型（如 Qwen-7B）的隐藏层维度
    )
)
def benchmark(num_tokens, hidden_size, provider):
    # 初始化数据
    x = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float16)
    residual = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float16)
    weight = torch.ones(hidden_size, device='cuda', dtype=torch.float16)
    eps = 1e-6

    # 预热并测量耗时 (毫秒)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms_norm_pytorch(x, residual, weight, eps), quantiles=quantiles
        )
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms_norm_fused(x, residual, weight, eps), quantiles=quantiles
        )

    # -------------------------------------------------------------
    # 计算理论显存访问量 (Bytes)
    # 读: x (2 bytes), residual (2 bytes)
    # 写: y (2 bytes), r_out (2 bytes)
    # 注: weight 的读取因为尺寸较小(被多行广播复用)可以忽略不计
    # 所以总访问量约为 4 个相同尺寸的张量
    # -------------------------------------------------------------
    gbps = lambda ms: (4 * x.numel() * x.element_size()) / (ms * 1e-3) / 1e9
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == '__main__':
    # print_data=True 会在终端输出类似 Markdown 表格的数据
    # save_path='.' 会把结果保存成 .png 图片
    benchmark.run(print_data=True, save_path='.')