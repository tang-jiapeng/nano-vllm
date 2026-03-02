import torch
from nanovllm.kernels.rms_norm import rms_norm_fused


def test_rms_norm_correctness():
    torch.manual_seed(42)
    B, H = 128, 4096
    # 初始化数据，使用 float16 模拟真实推理场景
    x = torch.randn(B, H, dtype=torch.float16, device="cuda")
    r = torch.randn(B, H, dtype=torch.float16, device="cuda")
    w = torch.ones(H, dtype=torch.float16, device="cuda")

    # ---------------- 1. PyTorch 参考实现 ----------------
    x_fp32 = x.float()
    r_fp32 = r.float()
    x_plus_r = x_fp32 + r_fp32

    rms = torch.sqrt((x_plus_r**2).mean(dim=-1, keepdim=True) + 1e-6)
    ref_y = (x_plus_r / rms * w.float()).half()
    ref_r = x_plus_r.half()

    # ---------------- 2. Triton 实现 ----------------
    tri_y, tri_r = rms_norm_fused(x, r, w)

    # ---------------- 3. 对比 ----------------
    # 允许一定的精度误差
    torch.testing.assert_close(tri_y, ref_y, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(tri_r, ref_r, atol=1e-3, rtol=1e-3)
    print("✅ 正确性验证通过！Triton 实现与 PyTorch 参考结果一致。")

if __name__ == "__main__":
    test_rms_norm_correctness()
