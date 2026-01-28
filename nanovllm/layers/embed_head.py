"""
词嵌入层和语言模型头实现

本模块实现了支持张量并行的词嵌入层和输出层：
1. VocabParallelEmbedding - 词嵌入层（token ID -> 向量）
2. ParallelLMHead - 语言模型头（向量 -> logits）

张量并行策略：
- 在词表维度上分片（vocab_size维度）
- 每个GPU持有部分词表
- 通过掩码和all-reduce处理跨GPU的词嵌入

使用场景：
- Token嵌入（将token ID转换为向量表示）
- 输出层（将隐藏状态转换为logits）
- 词表并行的分布式训练和推理

参考资料：
- Megatron-LM的词嵌入实现
- 张量并行的词汇分布策略
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
        词表并行嵌入层

        原理：
        - 在词表维度（vocab_size）上分片
        - 每个GPU持有部分词表（vocab_size / tp_size个词）
        - 通过掩码处理超出本地词表的token
        - 使用all-reduce聚合各GPU的嵌入结果

        内存布局：
        ```
        原始词嵌入：[vocab_size, embedding_dim]
        GPU 0: [vocab_size/tp_size, embedding_dim]  # 词0
        GPU 1: [vocab_size/tp_size, embedding_dim]  # 词1
        ...
        ```

        优势：
        - 内存效率：每个GPU只保存部分词嵌入
        - 可扩展：支持超大词表（百万级别）
        - 兼容：与张量并行的其他层无缝配合

        劣势：
    - 需要掩码和通信处理跨GPU词表
    - 索引操作可能影响性能

        使用场景：
    - Transformer的token嵌入层
    - 词表并行的分布式推理
    - 大词汇表模型（如ChatGLM、Qwen等）
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        """
        初始化词嵌入层。

        参数：
        - num_embeddings: 词表大小（vocab_size）
        - embedding_dim: 嵌入维度（每个token的向量维度）

        注意：
        num_embeddings必须能被tp_size整除
        """
        super().__init__()

        # 张量并行配置
        self.tp_rank = dist.get_rank()  # 当前GPU排名
        self.tp_size = dist.get_world_size()  # 总GPU数

        # 验证词表大小可被GPU数整除
        assert num_embeddings % self.tp_size == 0

        # 保存配置
        self.num_embeddings = num_embeddings  # 总词表大小
        self.num_embeddings_per_partition = (
            num_embeddings // self.tp_size
        )  # 每个GPU的词表大小

        # 计算当前GPU负责的词表范围
        self.vocab_start_idx = (
            self.num_embeddings_per_partition * self.tp_rank
        )  # 起始词ID
        self.vocab_end_idx = (
            self.vocab_start_idx + self.num_embeddings_per_partition
        )  # 结束词ID

        # 创建词嵌入权重（每个GPU只创建自己的部分）
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        # 设置权重加载器
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器

        从完整的词嵌入权重中切分出当前GPU应该持有的部分
        """
        param_data = param.data

        # 当前GPU应持有的分片大小
        shard_size = param_data.size(0)

        # 计算当前GPU的起始索引
        start_idx = self.tp_rank * shard_size

        # 从完整权重中切分出对应部分
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)

        # 复制到参数中
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        参数：
        - x: 输入token ID [batch_size, seq_len]

        返回值：
        - 输出嵌入 [batch_size, seq_len, embedding_dim]

        工作流程：
        1. 如果使用张量并行：
           - 创建掩码标记当前GPU负责的词
           - 调整token ID（减去起始偏移）
        2. 查表：F.embedding(x, self.weight)
        3. 如果使用张量并行：
           - 应用掩码
           - all-reduce聚合结果
        """
        # 步骤1：如果是张量并行，调整token ID
        if self.tp_size > 1:
            # 创建掩码：标记哪些token属于当前GPU
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)

            # 调整token ID：将全局ID转换为本地ID
            # 例如：GPU 0负责词0-999，本地ID = 全局ID - 0
            #      GPU 1负责词1000-1999，本地ID = 全局ID - 1000
            x = mask * (x - self.vocab_start_idx)

        # 步骤2：查表嵌入
        # F.embedding：查找token ID对应的嵌入向量
        y = F.embedding(x, self.weight)

        # 步骤3：如果使用张量并行，聚合结果
        if self.tp_size > 1:
            # 应用掩码：将不属于当前GPU的词对应的嵌入设为0
            y = mask.unsqueeze(1) * y

            # all-reduce：所有GPU的y相加
            # 聚合后，每个位置都有所有GPU贡献的嵌入向量之和
            dist.all_reduce(y)

        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型头

    用途：
    将隐藏状态转换为logits（对数几率），用于语言建模

    特点：
    - 继承VocabParallelEmbedding，支持词表并行
    - 不使用偏置（现代大语言模型的常见做法）
    - 支持prefill模式的特殊处理
    - 在张量并行时聚合各GPU的logits

    与普通线性层的区别：
    - 使用词表并行而非张量并行
    - 专门用于语言模型输出
    - 支持批量和prefill模式

    使用场景：
    - 语言模型的最终输出层
    - 计算每个词的概率分布
    - 文本生成的最后一步

    性能优化：
    - prefill模式：只计算最后一个位置的logits
    - 张量并行：gather操作聚合各GPU的logits
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        """
        初始化语言模型头。

        参数：
        - num_embeddings: 词表大小
        - embedding_dim: 隐藏层维度
        - bias: 是否使用偏置（默认不使用）

        注意：
        - 根据assert，现代大语言模型通常不使用偏置
        - 词表共享（weight tying）时，权重会在外部绑定
        """
        # 验证不使用偏置
        assert not bias

        # 调用父类初始化
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        参数：
        - x: 隐藏状态 [batch_size, seq_len, embedding_dim]

        返回值：
        - logits: 对数几率 [batch_size, seq_len, vocab_size]
                  或 [batch_size, vocab_size]（prefill模式）

        工作流程：
        1. 如果是prefill模式：
           - 只取最后一个位置的隐藏状态
           - 连续存储以优化性能
        2. 线性变换：logits = x @ W^T
        3. 如果使用张量并行：
           - gather各GPU的部分logits
           - 拼接得到完整logits
        """
        # 获取推理上下文
        context = get_context()

        # 步骤1：prefill模式的特殊处理
        if context.is_prefill:
            # 在prefill阶段，我们只需要最后一个token的logits进行采样
            # 获取每个序列的最后一个位置索引
            last_indices = context.cu_seqlens_q[1:] - 1

            # 提取最后一个位置的隐藏状态
            x = x[last_indices].contiguous()

        # 步骤2：线性变换
        # 从隐藏状态计算logits：logits = x @ W^T
        # 注意：不使用偏置（现代LLM的常见做法）
        logits = F.linear(x, self.weight)

        # 步骤3：张量并行的logits聚合
        if self.tp_size > 1:
            # 为gather操作准备接收缓冲区
            # 仅主GPU（rank 0）需要准备完整的接收缓冲区
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None

            # gather操作：收集所有GPU的logits
            dist.gather(logits, all_logits, 0)

            # 主GPU拼接各GPU的部分logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits
