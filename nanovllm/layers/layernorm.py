"""
RMSNorm（Root Mean Square Layer Normalization）实现

本模块实现了RMSNorm层归一化方法，是现代大语言模型（如LLaMA、Qwen等）的标准归一化方法。

RMSNorm vs LayerNorm：
1. RMSNorm更简单：去除了均值中心化步骤
2. 计算效率高：少一次减法和一次加法
3. 性能相当：在大多数任务上表现与LayerNorm相似
4. 数值稳定：RMS（均方根）天然为正，避免除零

数学公式：
RMSNorm: y = x * (1 / sqrt(mean(x^2) + eps)) * gamma
LayerNorm: y = (x - mean(x)) * (1 / sqrt(var(x) + eps)) * gamma

优势：
- 减少计算量：无需计算均值
- 减少内存带宽：少两次内存访问
- 保持效果：在归一化效果上与LayerNorm相当
- 梯度稳定：避免均值中心化引入的梯度问题

参考资料：
- RMSNorm论文：https://arxiv.org/abs/1910.07467
- LLaMA使用RMSNorm：https://arxiv.org/abs/2302.13971
- 代码参考：https://github.com/facebookresearch/llama/blob/main/llama/model.py
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    RMSNorm层归一化

    功能：
    对输入进行均方根归一化，稳定训练过程，加速收敛

    特点：
    1. 预归一化（Pre-Norm）：在子层之前应用归一化
    2. 残差连接支持：add_rms_forward支持原地残差操作
    3. 数值稳定：使用eps防止除零错误
    4. 性能优化：torch.compile加速计算

    使用场景：
    - Transformer解码器层（注意力前、FFN前）
    - 预归一化架构（Pre-Norm）
    - 现代大语言模型（Llama、Qwen、ChatGLM等）

    与LayerNorm的区别：
    - RMSNorm不进行均值中心化
    - 计算更简单，效率更高
    - 内存带宽需求更小
    - 在大模型上表现相当

    性能优势：
    - 减少2次算术运算（减法和加法）
    - 减少2次内存访问（读取均值和方差）
    - 编译优化：torch.compile进一步加速
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """
        初始化RMSNorm层。

        参数：
        - hidden_size: 归一化的维度大小
        - eps: 数值稳定性参数（默认：1e-6）

        注意：
        - weight是可学习的缩放参数（gamma）
        - 没有偏置参数（beta），这是RMSNorm的特点
        - eps应该足够小以保持归一化效果，但足够大以防止数值不稳定
        """
        super().__init__()

        # 保存数值稳定性参数
        self.eps = eps

        # 可学习的缩放参数（gamma）
        # 初始化为全1，不改变输入的尺度
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        标准RMSNorm前向传播

        参数：
        - x: 输入张量 [..., hidden_size]

        返回值：
        - 归一化后的张量，形状同x

        算法流程：
        1. 保存原始数据类型
        2. 转换为float32（提高计算精度）
        3. 计算RMS：sqrt(mean(x^2))
        4. 归一化：x / (RMS + eps)
        5. 应用缩放：* weight
        6. 转换回原始数据类型

        数学公式：
        ```
        rms = sqrt(mean(x^2))
        normalized = x / (rms + eps)
        output = normalized * weight
        ```

        优化技巧：
        - 原地操作：使用mul_()减少内存分配
        - rsqrt：计算1/sqrt(x)更高效
        - 保持数据类型：原始数据类型的精度可能较低（float16/bfloat16）
        - torch.compile：JIT编译加速计算

        使用示例：
        ```python
        norm = RMSNorm(hidden_size=4096)
        x = torch.randn(2, 8, 4096)  # [batch, seq, hidden]
        y = norm.rms_forward(x)  # [2, 8, 4096]
        ```
        """
        # 步骤1：保存原始数据类型
        # 归一化后需要转换回原始类型（如float16）
        orig_dtype = x.dtype

        # 步骤2：转换为float32
        # 提高计算精度，避免浮点误差
        x = x.float()

        # 步骤3：计算均方根（RMS）
        # var = mean(x^2)，然后取平方根
        # 使用keepdim保持维度，便于广播
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # 步骤4：归一化
        # x = x / sqrt(var + eps)
        # 使用rsqrt(x) = 1/sqrt(x)更高效
        x.mul_(torch.rsqrt(var + self.eps))

        # 步骤5：应用缩放参数
        # 输出 = 归一化后的x * weight
        # 先转换回原始数据类型，再乘以weight
        x = x.to(orig_dtype).mul_(self.weight)

        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        带残差连接的RMSNorm前向传播

        功能：
        将输入x与残差residual相加，然后进行RMSNorm归一化
        支持原地操作，减少内存分配

        参数：
        - x: 当前层的输出 [..., hidden_size]
        - residual: 残差连接（上一层的输出）[..., hidden_size]

        返回值：
        - tuple[归一化后的x, 更新后的residual]

        用途：
        在Pre-Norm架构中，将子层输出与残差相加，然后归一化
        例如：attention(x) + x -> LayerNorm -> output

        算法流程：
        1. 保存原始数据类型
        2. 残差相加：x = x + residual（原地操作）
        3. 转换回原始数据类型
        4. RMSNorm归一化
        5. 返回归一化后的x和新的residual

        优化技巧：
        - 原地相加：add_()减少内存分配
        - 数据类型转换：residual也需要转换以保持精度一致
        - 复用内存：新的residual直接使用x的内存

        使用示例：
        ```python
        norm = RMSNorm(hidden_size=4096)
        x = torch.randn(2, 8, 4096)  # 当前输出
        residual = torch.randn(2, 8, 4096)  # 残差

        # 原地相加后归一化
        normalized_x, new_residual = norm.add_rms_forward(x, residual)
        ```
        """
        # 步骤1：保存原始数据类型
        orig_dtype = x.dtype

        # 步骤2：残差相加（原地操作）
        # x.float().add_(residual.float()) 等价于：
        # x = x.float() + residual.float()
        x = x.float().add_(residual.float())

        # 步骤3：转换回原始数据类型
        # 新的residual使用x的内存，避免额外分配
        residual = x.to(orig_dtype)

        # 步骤4：RMSNorm归一化
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)

        # 注意：这里用residual接收归一化后的值
        # 最终返回：(normalized_x, new_residual)
        # normalized_x是归一化后的x（已包含残差）
        # new_residual也是相同的值（用于下一层）

        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（统一接口）

        参数：
        - x: 输入张量 [..., hidden_size]
        - residual: 残差连接（可选）[..., hidden_size]
                   None表示不使用残差
                   非None表示使用add_rms_forward

        返回值：
        - 如果residual为None：torch.Tensor [.., hidden_size]
        - 如果residual非None：tuple[torch.Tensor, torch.Tensor]
                              (归一化后的x, 新的residual)

        使用场景：
        1. 无残差：layer.norm(x)
           例如：输入层归一化、最终归一化

        2. 有残差：layer.norm(x, residual)
           例如：DecoderLayer中的pre-norm残差连接

        设计优势：
        - 统一接口：同一方法处理两种情况
        - 灵活性：支持有/无残差的场景
        - 内存效率：原地操作减少内存分配
        - 性能优化：根据条件选择不同的实现

        使用示例：
        ```python
        norm = RMSNorm(hidden_size=4096)

        # 场景1：无残差
        x1 = torch.randn(2, 8, 4096)
        y1 = norm(x1)  # 调用rms_forward

        # 场景2：有残差
        x2 = torch.randn(2, 8, 4096)
        residual = torch.randn(2, 8, 4096)
        y2, new_res = norm(x2, residual)  # 调用add_rms_forward
        ```
        """
        # 根据是否有残差选择不同的前向传播方法
        if residual is None:
            # 无残差：使用标准RMSNorm
            return self.rms_forward(x)
        else:
            # 有残差：使用带残差连接的RMSNorm
            return self.add_rms_forward(x, residual)
