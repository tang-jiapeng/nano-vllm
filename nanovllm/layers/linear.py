"""
张量并行线性层实现

本模块实现了支持张量并行的各种线性层，是nano-vllm的核心组件之一。
通过在多个GPU上分布计算，实现大模型的并行推理。

张量并行（Tensor Parallelism）原理：
- 将模型参数和计算分布到多个GPU上
- 每个GPU只持有部分参数
- 通过集体通信（all-reduce）聚合结果
- 适用于超大模型（单卡无法容纳）

支持的并行策略：
1. ReplicatedLinear - 复制（无并行）
2. ColumnParallelLinear - 列并行
3. RowParallelLinear - 行并行
4. MergedColumnParallelLinear - 合并列并行（多个输出）
5. QKVParallelLinear - QKV专用并行

参考资料：
- Megatron-LM: https://arxiv.org/abs/1909.08053
- NVIDIA Tensor Parallel: https://docs.nvidia.com/deeplearning/parallel-fusion/latest/user-guide/feature-parallel_fusion.html
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """
    安全除法，确保能够整除

    参数：
    - numerator: 被除数
    - denominator: 除数

    返回值：
    - 商（整数）

    用途：
    张量并行要求参数能够均匀分布到各个GPU上
    这个函数确保除法是整数除法，否则抛出异常

    使用示例：
    ```python
    # 正确用法
    divide(4096, 4)  # 返回 1024

    # 错误用法（会抛出异常）
    divide(4096, 3)  # AssertionError
    ```
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    线性层基类

    提供张量并行的通用基础设施：
    1. 张量并行配置（rank、size、维度）
    2. 权重参数初始化
    3. 偏置参数初始化
    4. 自定义权重加载器

    设计模式：
    - 使用weight_loader自定义权重加载逻辑
    - 支持张量并行的权重分片
    - 统一的参数管理接口
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        """
        初始化线性层基类。

        参数：
        - input_size: 输入维度
        - output_size: 输出维度
        - bias: 是否使用偏置
        - tp_dim: 张量并行的维度（0=列并行，1=行并行，None=无并行）

        张量并行配置：
        - tp_rank: 当前GPU的排名（0到tp_size-1）
        - tp_size: 总GPU数量
        - tp_dim: 并行维度（权重矩阵的哪个维度被分片）

        内存布局：
        - weight: [output_size, input_size]
        - bias: [output_size]（如果启用）
        """
        super().__init__()

        # 张量并行配置
        self.tp_dim = tp_dim  # 并行维度
        self.tp_rank = dist.get_rank()  # 当前GPU排名
        self.tp_size = dist.get_world_size()  # 总GPU数

        # 创建权重参数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 设置自定义权重加载器（用于张量并行）
        self.weight.weight_loader = self.weight_loader

        # 创建偏置参数（可选）
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 不使用偏置时，注册为None参数
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（子类实现）

        抛出NotImplementedError，由子类实现具体的计算逻辑
        """
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制线性层（无并行）

    特点：
    - 所有GPU持有完整的权重副本
    - 每个GPU独立计算完整结果
    - 无需GPU间通信
    - 适用于小模型或模型的小层

    内存使用：
    - 每个GPU都保存完整权重
    - 总内存 = tp_size * 权重大小

    使用场景：
    - 词嵌入层（embedding）
    - 最终输出层（lm_head）
    - 小于1B参数的层
    - 张量并行不可行时

    优势：
    - 实现简单
    - 无通信开销
    - 调试方便

    劣势：
    - 内存效率低
    - 无法处理超大模型
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """
        初始化复制线性层。

        参数：
        - input_size: 输入维度
        - output_size: 输出维度
        - bias: 是否使用偏置
        """
        # 无张量并行（tp_dim=None）
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器 - 简单复制

        将加载的权重直接复制到参数中
        所有GPU都获得完整的权重副本
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        标准线性变换：y = x @ W^T + b

        参数：
        - x: 输入张量 [..., input_size]

        返回值：
        - 输出张量 [..., output_size]
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层

    原理：
    - 将权重矩阵按列（输出维度）分片
    - 每个GPU持有部分列（output_size / tp_size）
    - 输入在所有GPU上完整存在
    - 输出在各GPU上部分存在
    - 无需聚合结果（各GPU已有最终结果）

    内存布局：
    ```
    原始权重：[output_size, input_size]
    GPU 0: [output_size/tp_size, input_size]  # 列0
    GPU 1: [output_size/tp_size, input_size]  # 列1
    ...
    ```

    计算流程：
    - 输入x在所有GPU上完整复制
    - 每个GPU计算 y_i = x @ W_i^T + b_i
    - 无通信（各GPU的输出不同部分）

    优势：
    - 内存效率高
    - 无通信开销
    - 适合QKV投影等场景

    劣势：
    - 输出需要后续聚合（如RowParallelLinear）
    - 仅适用于输出可分片的场景

    使用场景：
    - QKV投影
    - 门控投影
    - 上投影（up projection）
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """
        初始化列并行线性层。

        参数：
        - input_size: 输入维度
        - output_size: 输出维度（会被分片到各GPU）
        - bias: 是否使用偏置

        注意：
        output_size必须能被tp_size整除
        """
        tp_size = dist.get_world_size()
        # 输出维度按GPU数分片
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器 - 列分片

        从完整的权重中切分出当前GPU应该持有的部分
        """
        param_data = param.data

        # 当前GPU应持有的分片大小
        shard_size = param_data.size(self.tp_dim)

        # 计算当前GPU的起始索引
        start_idx = self.tp_rank * shard_size

        # 从完整权重中切分出对应部分
        # narrow(dim, start, length)在指定维度上切片
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)

        # 复制到参数中
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        标准线性变换，每个GPU只计算部分输出
        """
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层

    用途：
    同时处理多个线性投影，将它们合并为一个层
    例如：SwiGLU中的gate和up投影

    原理：
    - 将多个投影的权重在列维度上拼接
    - 输入一个，输出多个
    - 每个GPU持有所有投影的部分列

    内存布局（两个投影的例子）：
    ```
    原始权重：
    - gate_proj: [intermediate_size, input_size]
    - up_proj: [intermediate_size, input_size]
    合并后: [2*intermediate_size, input_size]

    GPU分布（2个GPU）：
    - GPU 0: [intermediate_size, input_size]  # gate的第0半 + up的第0半
    - GPU 1: [intermediate_size, input_size]  # gate的第1半 + up的第1半
    ```

    优势：
    - 减少一次矩阵乘法
    - 提高内存访问效率
    - 适合并行投影场景

    使用场景：
    - SwiGLU的gate和up投影
    - 多头注意力的QKV投影
    - 需要多个并行投影的场景
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """
        初始化合并列并行线性层。

        参数：
        - input_size: 输入维度
        - output_sizes: 各个输出的大小列表
                       例如：[intermediate_size, intermediate_size] for gate and up
        - bias: 是否使用偏置

        注意：
        output_sizes中所有值都必须能被tp_size整除
        """
        # 保存各个输出的大小
        self.output_sizes = output_sizes

        # 总输出大小是所有大小的和
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ):
        """
        权重加载器 - 合并列分片

        参数：
        - loaded_shard_id: 当前加载的投影ID
                          例如：0表示gate，1表示up

        工作流程：
        1. 计算该投影在合并权重中的起始偏移
        2. 计算该投影的分片大小
        3. 切分并加载对应部分
        """
        param_data = param.data

        # 计算当前投影的起始偏移（所有之前投影的总和）
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size

        # 当前投影的分片大小
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        # 在参数中切分出对应区域
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        # 将完整权重分片到各GPU
        # chunk返回分片列表，选择当前GPU的分片
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        # 复制数据
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV并行线性层

    专门用于多头注意力中的Q、K、V投影
    将三个投影合并为一个层，提高效率

    特点：
    1. 支持Grouped Query Attention (GQA)
       - Q的头数可以与K、V不同
       - K、V使用较少的头数
    2. 内存优化
       - 一次矩阵乘法得到Q、K、V
       - 减少内存访问次数
    3. 支持张量并行
       - Q、K、V分别分片

    权重布局：
    ```
    原始权重：[total_num_heads + 2*total_num_kv_heads, head_size, input_size]

    分片后（tp_size=2）：
    - GPU 0: [num_heads/2 + num_kv_heads, head_size, input_size]
    - GPU 1: [num_heads/2 + num_kv_heads, head_size, input_size]
    ```

    使用场景：
    - Transformer的注意力层
    - 替换三个独立的线性层（q_proj, k_proj, v_proj）
    - 支持GQA优化
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """
        初始化QKV并行线性层。

        参数：
        - hidden_size: 隐藏层维度（输入）
        - head_size: 每个头的维度
        - total_num_heads: Q的总头数
        - total_num_kv_heads: K、V的总头数（默认等于total_num_heads）
        - bias: 是否使用偏置

        注意：
        - total_num_heads和total_num_kv_heads都必须能被tp_size整除
        - 通常total_num_kv_heads <= total_num_heads（GQA）
        """
        tp_size = dist.get_world_size()

        # K、V头数默认为Q头数
        total_num_kv_heads = total_num_kv_heads or total_num_heads

        # 保存配置
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)  # 每个GPU的Q头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # 每个GPU的K、V头数

        # 总输出大小：Q头数 + K头数 + V头数） * 每头维度
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size

        # 调用父类初始化
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ):
        """
        权重加载器 - QKV分片

        参数：
        - loaded_shard_id: 标识加载的投影类型
                          "q" = Q投影
                          "k" = K投影
                          "v" = V投影

        工作流程：
        1. 根据loaded_shard_id确定投影类型
        2. 计算该投影在合并权重中的偏移
        3. 计算该投影的分片大小
        4. 切分并加载对应部分
        """
        param_data = param.data

        # 验证投影ID
        assert loaded_shard_id in ["q", "k", "v"]

        # 根据投影类型计算偏移和大小
        if loaded_shard_id == "q":
            # Q投影在最开始
            shard_size = self.num_heads * self.head_size  # Q头数 * 每头维度
            shard_offset = 0  # 起始偏移为0
        elif loaded_shard_id == "k":
            # K投影在Q之后
            shard_size = self.num_kv_heads * self.head_size  # KV头数 * 每头维度
            shard_offset = self.num_heads * self.head_size  # Q之后
        else:  # loaded_shard_id == "v"
            # V投影在Q和K之后
            shard_size = self.num_kv_heads * self.head_size  # KV头数 * 每头维度
            shard_offset = (
                self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            )  # Q和K之后

        # 在参数中切分出对应区域
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)

        # 分片加载
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]

        # 复制数据
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
        行并行线性层

        原理：
        - 将权重矩阵按行（输入维度）分片
        - 每个GPU持有部分行（input_size / tp_size）
        - 输入在各GPU上部分存在
        - 输出在所有GPU上完整存在
        - 需要all-reduce聚合结果

        内存布局：
        ```
        原始权重：[output_size, input_size]
        GPU 0: [output_size, input_size/tp_size]  # 行0
        GPU 1: [output_size, input_size/tp_size]  # 行1
        ...
        ```

        计算流程：
        1. 输入x在第1维（输入维度）被分片到各GPU
        2. 每个GPU计算 y_i = x_i @ W_i^T
        3. 使用all-reduce聚合所有GPU的输出
        4. 得到完整输出 y = sum_i y_i

        优势：
        - 内存效率高
        - 支持任意输出大小

        劣势：
    - 需要通信（all-reduce）
    - 通信开销与输出大小成正比

        使用场景：
        - 注意力输出投影（o_proj）
        - FFN下投影（down_proj）
        - 输出层前的投影

        通信优化：
        - 使用NVIDIA NCCL或MPI进行高效通信
        - 支持异步all-reduce
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """
        初始化行并行线性层。

        参数：
        - input_size: 输入维度（会被分片到各GPU）
        - output_size: 输出维度
        - bias: 是否使用偏置

        注意：
        input_size必须能被tp_size整除
        """
        tp_size = dist.get_world_size()

        # 输入维度按GPU数分片
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)
        
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器 - 行分片

        从完整的权重中切分出当前GPU应该持有的部分
        """
        param_data = param.data

        # 当前GPU应持有的分片大小
        shard_size = param_data.size(self.tp_dim)

        # 计算当前GPU的起始索引
        start_idx = self.tp_rank * shard_size

        # 从完整权重中切分出对应部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)

        # 复制到参数中
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        计算流程：
        1. 线性变换：y = x @ W^T
        2. 偏置：y = y + b（仅主GPU）
        3. all-reduce：聚合所有GPU的输出
        4. 返回完整输出
        """
        # 第1步：线性变换
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        # 第2步：通信聚合（如果使用张量并行）
        if self.tp_size > 1:
            # all-reduce：所有GPU的y相加
            dist.all_reduce(y)

        return y
