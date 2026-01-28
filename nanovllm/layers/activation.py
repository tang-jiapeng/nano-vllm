import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SwiGLU激活函数。

    SwiGLU公式：
    silu(x) * y
    其中：
    - x, y是输入张量的两半
    - silu(x) = x * sigmoid(x)

    使用场景：
    - Qwen模型的FFN层
    - 现代transformer的激活函数
    - 相比ReLU有更好的性能

    优点：
    - 平滑的非线性
    - 减少梯度消失
    - 性能优于ReLU、GELU等
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
        - x: 输入张量，最后一个维度是2*intermediate_size

        返回值：
        - SwiGLU激活后的张量

        处理流程：
        1. 将输入在最后一个维度上分成两半
        2. 对第一半应用SiLU激活
        3. 乘以第二半
        """
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
