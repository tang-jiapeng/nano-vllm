"""
Token采样器 - 从logits中采样下一个token

本模块实现了基于Gumbel-Max技巧的温度采样器，用于从模型输出的logits中采样token。
这是大语言模型推理时的关键组件，负责将概率分布转换为具体的token选择。

核心算法：
1. 温度缩放 - 控制输出分布的锐度
2. Gumbel-Max采样 - 实现可微的随机采样

使用场景：
- 自回归文本生成
- 批量推理服务
- 采样策略实现（temperature、top-k、top-p等）

参考论文：
- Gumbel-Max Trick: https://arxiv.org/abs/1611.01144
- Categorical Reparameterization: https://arxiv.org/abs/1611.01144
"""

import torch
from torch import nn


class Sampler(nn.Module):
    """
    基于Gumbel-Max技巧的token采样器

    功能特点：
    1. Gumbel-Max采样：实现可微的随机采样，保持梯度流
    2. 温度控制：支持每个序列不同的温度参数
    3. 高效实现：无需排序，直接采样
    4. 数值稳定：防止除零错误和数值溢出

    Gumbel-Max算法：
    1. 为每个类别i生成Gumbel噪声：g_i = -log(-log(U_i)), U_i ~ Uniform(0,1)
    2. 计算调整后的logits：logits_i / temperature + g_i
    3. 选择最大值对应的类别

    数学公式：
    ```
    g_i = -log(-log(U_i)), U_i ~ Uniform(0,1)
    adjusted_logits_i = logits_i / temperature + g_i
    sampled_token = argmax_i(adjusted_logits_i)
    ```

    温度效果：
    - 温度 = 1.0：保持原始分布
    - 温度 < 1.0：分布更尖锐，倾向于高概率token
    - 温度 > 1.0：分布更平坦，增加随机性
    - 温度 → 0：近似贪婪解码（总是选择最高概率token）

    与其他采样方法的比较：
    - 贪婪解码（temperature=0）：确定性，但缺乏创造性
    - Top-K：限制候选token数量，可能丢失长尾优质token
    - Top-P（nucleus）：动态候选集，但实现复杂
    - Temperature采样：简单灵活，是最基础的采样方法

    使用场景：
    - decode阶段从logits采样下一个token
    - 支持批量采样，每个序列可以有不同的温度
    - 可与其他策略结合（temperature + top-k等）
    """

    def __init__(self):
        """
        初始化采样器

        无需额外参数，所有配置都在forward方法中传入
        """
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        前向传播：执行token采样

        使用方法：
        ```python
        sampler = Sampler()
        tokens = sampler(logits, temperatures)
        ```

        参数：
        - logits: 模型输出的对数几率 [batch_size, vocab_size]
                  logits[i][j]表示第i个序列中token j的对数几率
        - temperatures: 每个序列的温度参数 [batch_size]
                        温度为标量，用于控制采样的随机性

        返回值：
        - sampled_tokens: 采样的token IDs [batch_size]
                         每个序列对应一个采样到的token

        算法详解：
        步骤1：温度缩放
               logits_scaled = logits / temperatures.unsqueeze(1)
               - 将每个序列的logits除以该序列的温度
               - temperatures形状为[batch_size]，需要unsqueeze为[batch_size, 1]
               - 广播后：每个vocab_size维度都除以相同的温度

        步骤2：计算概率分布
               probs = softmax(logits_scaled, dim=-1)
               - 将logits转换为概率分布
               - dim=-1：在vocab_size维度上计算softmax
               - 数值稳定性：使用log-sum-exp技巧

        步骤3：Gumbel-Max采样
               sample_tokens = argmax(log(probs) + gumbel_noise)
               - 技巧：probs / exponential(1) 相当于添加Gumbel噪声
               - exponential(1)：指数分布，参数λ=1
               - clamp_min_(1e-10)：防止除零错误
               - argmax：选择最大的调整后概率对应的token

        数值优化：
        1. div_()：原地除法，减少内存分配
        2. exponential_(1)：原地生成指数噪声
        3. clamp_min_：原地截断，防止极小值
        4. torch.compile：JIT编译加速计算

        使用示例：
        ```python
        sampler = Sampler()

        # 单序列采样
        logits = torch.randn(1, 10000)  # 1个序列，10000个词表
        temperature = torch.tensor([0.8])  # 温度0.8
        token = sampler(logits, temperature)  # [1]

        # 批量采样（不同温度）
        batch_logits = torch.randn(4, 10000)  # 4个序列
        temperatures = torch.tensor([0.5, 0.8, 1.0, 1.5])  # 不同温度
        tokens = sampler(batch_logits, temperatures)  # [4]

        # 常见温度值及其效果
        temperatures = torch.tensor([0.0])  # 贪婪解码（确定性）
        temperatures = torch.tensor([0.7])  # 较低温度（保守）
        temperatures = torch.tensor([1.0])  # 标准温度（平衡）
        temperatures = torch.tensor([1.5])  # 较高温度（随机）
        ```

        注意事项：
        - logits应该是未标准化的对数几率（不要先softmax）
        - 温度应该为正数（> 0）
        - 温度=0时，行为类似贪婪解码
        - 批量处理时，每个序列可以有不同的温度
        - 返回的是token IDs，需要通过tokenizer转换为文本
        """
        # 步骤1：温度缩放
        # 将logits除以温度进行缩放，控制分布的锐度
        # temperatures形状：[batch_size] -> [batch_size, 1]
        # 广播后：[batch_size, vocab_size]
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 步骤2：计算概率分布
        # 将logits转换为概率分布（和为1）
        probs = torch.softmax(logits, dim=-1)

        # 步骤3：Gumbel-Max采样
        # 技巧：probs / exponential(1) 相当于添加Gumbel噪声
        # 公式：g = -log(-log(U)), U ~ Uniform(0,1)
        # 等价于：probs / exponential(1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        return sample_tokens
