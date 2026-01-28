"""
Qwen3模型实现 - 专门为Qwen3系列模型优化的推理引擎组件

本模块实现了Qwen3模型的核心组件，包括：
1. Qwen3Attention - 带RoPE的多头注意力机制
2. Qwen3MLP - SwiGLU激活的前馈网络
3. Qwen3DecoderLayer - 单个解码器层
4. Qwen3Model - 完整的transformer模型
5. Qwen3ForCausalLM - 因果语言模型（带输出头）

特点：
- 支持张量并行（tensor parallelism）
- 集成RoPE（旋转位置编码）
- 优化的KV-cache管理
- 支持Grouped-Query Attention (GQA)
"""

import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3的多头注意力机制（带RoPE和GQA支持）

    功能特点：
    1. 支持Grouped-Query Attention (GQA) - K和V使用较少的注意力头
    2. 集成RoPE（旋转位置编码）进行位置信息编码
    3. 优化的QKV投影（使用QKVParallelLinear减少内存访问）
    4. 输出投影使用RowParallelLinear实现张量并行
    5. Q和K使用RMSNorm进行归一化（提升稳定性）

    使用场景：
    - 适用于长序列处理（max_position可配置）
    - 支持张量并行推理
    - 兼容prefix caching优化

    内存布局：
    - Q, K, V的形状：[batch_size, num_heads, head_dim]
    - 输出形状：[batch_size, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        """
        初始化Qwen3Attention层。

        参数说明：
        - hidden_size: 模型的隐藏层维度（例如：4096）
        - num_heads: Q的注意力头数量（例如：32）
        - num_kv_heads: K和V的注意力头数量（用于GQA，例如：8）
        - max_position: 最大序列长度（默认：4096*32=131072）
        - head_dim: 每个头的维度（默认：hidden_size // num_heads）
        - rms_norm_eps: RMSNorm的数值稳定性参数（默认：1e-6）
        - qkv_bias: 是否在QKV投影中使用偏置（默认：False）
        - rope_theta: RoPE的基础频率（默认：10000）
        - rope_scaling: RoPE的缩放配置（默认：None）

        张量并行处理：
        - 将num_heads和num_kv_heads平均分配到各个GPU
        - 每个GPU处理num_heads // tp_size个Q头
        - 每个GPU处理num_kv_heads // tp_size个K、V头

        组件说明：
        1. qkv_proj: 线性层，将hidden_size投影到Q、K、V的组合空间
        2. o_proj: 注意力输出投影层，从num_heads*head_dim回到hidden_size
        3. rotary_emb: RoPE编码器，为Q和K添加位置信息
        4. attn: 注意力计算核心（使用flash-attention）
        5. q_norm/k_norm: 对Q和K进行RMSNorm归一化

        使用示例：
        ```python
        attn = Qwen3Attention(
            hidden_size=4096,
            num_heads=32,
            num_kv_heads=8,  # GQA，K和V使用8个头
            max_position=131072,
            head_dim=128,
            rope_theta=10000,
        )
        ```
        """
        super().__init__()
        # 获取张量并行的世界大小（GPU数量）
        tp_size = dist.get_world_size()

        # 总的注意力头数量
        self.total_num_heads = num_heads
        # 检查是否能被GPU数量整除（张量并行的要求）
        assert self.total_num_heads % tp_size == 0
        # 当前GPU处理的注意力头数量
        self.num_heads = self.total_num_heads // tp_size

        # 总的KV头数量（用于GQA）
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        # 当前GPU处理的KV头数量
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        # 计算每个头的维度
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # Q的大小 = 头数 * 每头维度
        self.q_size = self.num_heads * self.head_dim
        # KV的大小 = 头数 * 每头维度
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力缩放因子（防止softmax饱和）
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # 1. QKV投影层：hidden_size -> Q + K + V
        self.qkv_proj = QKVParallelLinear(
            hidden_size,  # 输入维度
            self.head_dim,  # 每个头的维度
            self.total_num_heads,  # Q的总头数
            self.total_num_kv_heads,  # K、V的总头数
            bias=qkv_bias,
        )

        # 2. 注意力输出投影层：num_heads*head_dim -> hidden_size
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,
        )

        # 3. RoPE编码器（旋转位置编码）
        self.rotary_emb = get_rope(
            self.head_dim,  # 旋转的维度
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        # 4. 注意力计算核心
        self.attn = Attention(
            self.num_heads,  # 当前GPU的Q头数
            self.head_dim,  # 每头的维度
            self.scaling,  # 缩放因子
            self.num_kv_heads,  # 当前GPU的KV头数
        )

        # 5. Q和K的RMSNorm归一化（提升训练稳定性）
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播计算。

        参数：
        - positions: 位置索引张量 [batch_size, seq_len]，表示每个token在序列中的位置
        - hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]，输入的token嵌入

        返回值：
        - 输出张量 [batch_size, seq_len, hidden_size]，注意力计算后的结果

        计算流程：
        1. QKV投影：将hidden_states通过线性层得到Q、K、V
        2. 形状变换：将Q、K、V重塑为 [batch_size, num_heads, head_dim] 的形状
        3. RMSNorm归一化：对Q和K分别进行RMSNorm归一化
        4. RoPE编码：为Q和K添加旋转位置信息
        5. 注意力计算：使用flash-attention计算注意力输出
        6. 输出投影：将注意力输出投影回hidden_size维度

        数据流示例（单序列）：
        输入：
        - positions: [0, 1, 2, 3] (4个token的位置)
        - hidden_states: [4, 4096] (4个token，每个4096维)

        内部计算：
        1. QKV投影：[4, 4096] -> [4, (32+8+8)*128] = [4, 6144]
        2. 切分Q、K、V：
           - Q: [4, 32*128] = [4, 4096]
           - K: [4, 8*128] = [4, 1024]
           - V: [4, 8*128] = [4, 1024]
        3. 重塑形状：
           - Q: [4, 32, 128]
           - K: [4, 8, 128]
           - V: [4, 8, 128]
        4. RoPE编码：保持形状不变，但Q和K的值发生旋转
        5. 注意力计算：[4, 32, 128] (GQA自动处理K、V头数差异)
        6. 输出投影：[4, 32*128] -> [4, 4096]

        使用示例：
        ```python
        attn = Qwen3Attention(...)
        positions = torch.tensor([[0, 1, 2], [0, 1, 2]])  # 2个序列，每个3个token
        hidden_states = torch.randn(2, 3, 4096)
        output = attn(positions, hidden_states)  # [2, 3, 4096]
        ```
        """
        # 第1步：QKV投影 - 将隐藏状态投影到Q、K、V空间
        qkv = self.qkv_proj(hidden_states)

        # 第2步：切分Q、K、V
        # qkv形状：[batch_size, seq_len, q_size + kv_size + kv_size]
        # 切分后：
        # - q: [batch_size, seq_len, q_size]
        # - k: [batch_size, seq_len, kv_size]
        # - v: [batch_size, seq_len, kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 第3步：重塑Q的形状 - 为多头注意力做准备
        # 从 [batch_size, seq_len, num_heads * head_dim]
        # 到   [batch_size, seq_len, num_heads, head_dim]
        # 然后转置为 [batch_size, num_heads, seq_len, head_dim]
        # 但这里保持 [batch_size * seq_len, num_heads, head_dim] 形状
        q = q.view(-1, self.num_heads, self.head_dim)

        # 第4步：重塑K和V的形状（类似Q）
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 第5步：应用RoPE编码 - 为Q和K添加位置信息
        # positions: [batch_size, seq_len] -> 广播到每个头
        # q, k: [batch_size * seq_len, num_heads, head_dim]
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        # 第6步：注意力计算 - 使用flash-attention高效计算
        # 输入：
        # - q: [batch_size * seq_len, num_heads, head_dim]
        # - k: [batch_size * seq_len, num_kv_heads, head_dim]
        # - v: [batch_size * seq_len, num_kv_heads, head_dim]
        # 输出：
        # - o: [batch_size * seq_len, num_heads, head_dim]
        o = self.attn(q, k, v)

        # 第7步：输出投影 - 将多头注意力的结果投影回hidden_size
        # 首先展平多头：[num_heads, head_dim] -> [num_heads * head_dim]
        # 然后通过RowParallelLinear投影
        output = self.o_proj(o.flatten(1, -1))

        # 返回：[batch_size, seq_len, hidden_size]
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3的前馈网络（Feed-Forward Network, FFN）

    实现了SwiGLU激活函数的前馈网络，是Transformer解码器的重要组成部分。

    功能特点：
    1. 使用SwiGLU激活函数（SiLU * Gate），性能优于ReLU和GELU
    2. 采用"gated linear unit"架构，通过门控机制控制信息流
    3. 支持张量并行：gate_up_proj使用MergedColumnParallelLinear，
       down_proj使用RowParallelLinear
    4. 三层结构：hidden_size -> 2*intermediate_size -> hidden_size

    SwiGLU优势：
    - 平滑的非线性，减少梯度消失
    - 门控机制提供稀疏性，提升模型表达能力
    - 在大语言模型中表现优于传统激活函数

    计算流程：
    x [hidden_size] -> gate_up_proj [2*intermediate_size] -> split [gate, up]
                      -> SiLU(gate) * up [intermediate_size]
                      -> down_proj [hidden_size]

    使用场景：
    - Transformer解码器的FFN层
    - 位于注意力层之后，用于特征变换
    - 在每个解码器层中占主导参数量（通常2/3的参数量）
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        """
        初始化Qwen3MLP。

        参数：
        - hidden_size: 输入和输出的隐藏层维度（如：4096）
        - intermediate_size: FFN中间层的维度（通常为hidden_size的2-4倍，如：11008）
        - hidden_act: 激活函数类型（应为"silu"）

        架构说明：
        1. gate_up_proj: 合并的线性层，同时计算门控和上投影
           - 输入：hidden_size
           - 输出：[intermediate_size, intermediate_size]（两路）
        2. down_proj: 下投影线性层
           - 输入：intermediate_size
           - 输出：hidden_size
        3. act_fn: SwiGLU激活函数

        使用示例：
        ```python
        mlp = Qwen3MLP(
            hidden_size=4096,
            intermediate_size=11008,  # 通常为2.7倍hidden_size
            hidden_act="silu"
        )
        x = torch.randn(2, 3, 4096)  # [batch_size, seq_len, hidden_size]
        output = mlp(x)  # [2, 3, 4096]
        ```
        """
        super().__init__()

        # 1. Gate和Up投影的合并线性层
        # 功能：将hidden_size投影到2*intermediate_size，然后切分为两路
        # 优势：减少一次矩阵乘法的内存访问
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,  # 输入维度
            [intermediate_size] * 2,  # 输出维度列表：[intermediate_size, intermediate_size]
            bias=False,
        )

        # 2. 下投影层
        # 功能：将intermediate_size投影回hidden_size
        # 使用RowParallelLinear支持张量并行
        self.down_proj = RowParallelLinear(
            intermediate_size,  # 输入维度
            hidden_size,  # 输出维度
            bias=False,
        )

        # 验证激活函数类型
        assert hidden_act == "silu"

        # 3. SwiGLU激活函数
        self.act_fn = SiluAndMul()

    def forward(self, x):
        """
        前向传播计算。

        参数：
        - x: 输入张量 [batch_size, seq_len, hidden_size]

        返回值：
        - 输出张量 [batch_size, seq_len, hidden_size]

        计算流程详解：
        1. gate_up投影：
           输入：[batch_size, seq_len, hidden_size]
           输出：[batch_size, seq_len, 2*intermediate_size]
        2. 切分Gate和Up：
           - gate: [batch_size, seq_len, intermediate_size]
           - up:   [batch_size, seq_len, intermediate_size]
        3. SwiGLU激活：
           activated = SiLU(gate) * up
           形状：[batch_size, seq_len, intermediate_size]
        4. 下投影：
           输出：[batch_size, seq_len, hidden_size]

        数学公式：
        ```
        gate_up = x @ W_gate_up^T + b_gate_up
        gate, up = split(gate_up, dim=-1)
        activated = SiLU(gate) * up
        output = activated @ W_down^T + b_down
        ```

        内存和计算优化：
        - 使用MergedColumnParallelLinear减少一次矩阵乘法
        - 激活函数直接作用于切分后的两路，避免额外的copy
        - 支持张量并行，在多个GPU上并行计算

        使用示例：
        ```python
        mlp = Qwen3MLP(hidden_size=4096, intermediate_size=11008, hidden_act="silu")
        x = torch.randn(4, 8, 4096)  # batch=4, seq_len=8
        y = mlp(x)  # [4, 8, 4096]
        ```
        """
        # 第1步：Gate和Up投影
        # 输入x通过合并的线性层得到gate和up的组合
        gate_up = self.gate_up_proj(x)  # [batch, seq_len, 2*intermediate_size]

        # 第2步：SwiGLU激活
        # SiluAndMul内部会切分输入，对前半部分应用SiLU，然后乘以后半部分
        x = self.act_fn(gate_up)  # [batch, seq_len, intermediate_size]

        # 第3步：下投影
        # 将中间维度投影回hidden_size
        x = self.down_proj(x)  # [batch, seq_len, hidden_size]

        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 Transformer的单个解码器层

    实现了标准的Transformer解码器层结构，包含：
    1. 多头自注意力机制（Self-Attention）
    2. 前馈网络（Feed-Forward Network, FFN）
    3. 残差连接和层归一化（Pre-Norm架构）

    架构特点：
    - 使用Pre-Norm而非Post-Norm：LayerNorm在子层之前应用
    - 支持残差连接：避免梯度消失，加速训练
    - 集成RoPE位置编码
    - 支持张量并行推理

    层次结构：
    输入 -> LayerNorm -> Self-Attention -> [残差连接]
                |                       |
                v                       v
           LayerNorm ----------------> FFN -> [残差连接] -> 输出

    优势（Pre-Norm vs Post-Norm）：
    - 训练更稳定，梯度流动更好
    - 减少训练初期的梯度爆炸问题
    - 允许更深层的网络结构
    - LLaMA、Qwen等现代大模型广泛采用

    使用场景：
    - Transformer解码器的标准层
    - 可堆叠多个DecoderLayer形成完整模型
    - 支持批量推理和张量并行
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        初始化单个解码器层

        参数：
        - config: Qwen3Config，模型的配置对象

        组件说明：
        1. self_attn: 自注意力层（带RoPE和GQA）
        2. mlp: 前馈网络（SwiGLU激活）
        3. input_layernorm: 注意力前的层归一化
        4. post_attention_layernorm: 注意力后的层归一化（FFN前）

        配置映射：
        - config.hidden_size -> hidden_size（输入/输出维度）
        - config.num_attention_heads -> num_heads（注意力头数）
        - config.num_key_value_heads -> num_kv_heads（KV头数，用于GQA）
        - config.max_position_embeddings -> max_position（最大位置）
        - config.rms_norm_eps -> eps（RMSNorm数值稳定性）
        - config.intermediate_size -> intermediate_size（FFN中间层维度）
        - config.hidden_act -> hidden_act（激活函数，应为"silu"）

        使用示例：
        ```python
        from transformers import Qwen3Config
        config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
        layer = Qwen3DecoderLayer(config)
        ```
        """
        super().__init__()

        # 1. 自注意力层（带RoPE位置编码）
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # 2. 前馈网络（SwiGLU激活）
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        # 3. 输入层归一化（注意力前）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 4. 注意力后层归一化（FFN前）
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算（Pre-Norm架构）。

        参数：
        - positions: 位置索引 [batch_size, seq_len]
        - hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
        - residual: 残差连接的上一层的输出（可选）
                   第一次调用时为None，后续调用时传入上一次的输出

        返回值：
        - hidden_states: 当前层的输出 [batch_size, seq_len, hidden_size]
        - residual: 残差（用于下一层），形状同hidden_states

        Pre-Norm计算流程：
        ```
        # 注意力块
        normed_x = LayerNorm(x)
        attn_out = SelfAttention(normed_x)
        x = x + attn_out  # 残差连接

        # FFN块
        normed_x = LayerNorm(x)
        ffn_out = MLP(normed_x)
        x = x + ffn_out   # 残差连接
        ```

        详细步骤：
        1. 输入层归一化并保存残差
           - residual = hidden_states (原始输入)
           - hidden_states = LayerNorm(hidden_states)
        2. 自注意力计算
           - hidden_states = SelfAttention(hidden_states, positions)
        3. 注意力后的层归一化
           - hidden_states = LayerNorm(hidden_states, residual)
           - residual = hidden_states (更新残差)
        4. FFN计算
           - hidden_states = MLP(hidden_states)
        5. 返回结果

        优势说明：
        - 残差连接：hidden_states包含原始输入信息，避免信息丢失
        - LayerNorm在子层前：稳定训练，防止梯度问题
        - 原地操作：RMSNorm的add_rms_forward减少内存分配

        内存优化：
        - 使用原地操作（in-place operation）减少内存分配
        - residual在层间传递，避免重复计算
        - 支持张量并行，多GPU协同计算

        使用示例：
        ```python
        layer = Qwen3DecoderLayer(config)
        positions = torch.tensor([[0, 1, 2], [0, 1, 2]])  # 2个序列
        hidden_states = torch.randn(2, 3, 4096)
        # 第一次调用（第一层）
        hidden_states, residual = layer(positions, hidden_states, None)
        # 后续调用（后续层）
        hidden_states, residual = layer(positions, hidden_states, residual)
        ```
        """
        # ========== 注意力块（Attention Block）==========

        # 步骤1：输入层归一化 + 残差保存
        # 注意：使用原地操作，residual保存原始hidden_states
        if residual is None:
            # 第一次调用（模型第一层）
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续调用，使用add_rms_forward原地更新
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # 步骤2：自注意力计算
        # 应用RoPE位置编码并计算注意力
        hidden_states = self.self_attn(positions, hidden_states)

        # ========== 前馈网络块（FFN Block）==========

        # 步骤3：注意力后层归一化 + 残差更新
        # 将attention输出与残差相加，然后归一化
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # 步骤4：FFN计算
        hidden_states = self.mlp(hidden_states)

        # 返回：当前输出 + 残差（传递给下一层）
        # residual一定不为None（要么是初始输入，要么是上一层的输出）
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3完整的Transformer模型（不含输出层）

    实现了完整的Transformer解码器结构，包含：
    1. 词嵌入层（Token Embedding）
    2. 多个解码器层（Decoder Layers）
    3. 最终的层归一化（Final LayerNorm）

    架构特点：
    - 堆叠N个Qwen3DecoderLayer
    - Pre-Norm架构：每层前都有LayerNorm
    - 最终输出经过RMSNorm
    - 支持张量并行推理
    - 兼容prefix caching

    层次结构：
    输入 -> Embedding -> DecoderLayer_1 -> ... -> DecoderLayer_N -> Final Norm -> 输出

    使用场景：
    - 作为Qwen3ForCausalLM的内部模型
    - 提取hidden states用于其他任务
    - 支持批量推理和KV-cache

    注意事项：
    - 不包含最终的输出投影层（lm_head）
    - 需要外部的compute_logits计算最终logits
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        """
        初始化Qwen3Model。

        参数：
        - config: Qwen3Config，模型配置

        组件说明：
        1. embed_tokens: 词嵌入层，将token ID转换为向量
        2. layers: 多个解码器层的列表（深度：config.num_hidden_layers）
        3. norm: 最终层归一化（RMSNorm）

        模型深度：
        - Qwen3-0.6B: 28层
        - Qwen3-1.5B: 28层
        - Qwen3-3B: 36层
        - Qwen3-7B: 32层
        - Qwen3-14B: 40层
        - Qwen3-32B: 48层

        内存和计算：
        - 参数量：≈ (embed + layers*N + norm)
        - 其中layers占大部分（每层约2*hidden_size^2参数）
        - 张量并行时参数均匀分布到各GPU

        使用示例：
        ```python
        from transformers import Qwen3Config
        config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
        model = Qwen3Model(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 10))
        positions = torch.arange(10).unsqueeze(0).repeat(2, 1)
        hidden_states = model(input_ids, positions)  # [2, 10, hidden_size]
        ```
        """
        super().__init__()

        # 1. 词嵌入层 - 将token ID映射为向量
        # 使用VocabParallelEmbedding支持张量并行
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        # 2. 多个解码器层 - 堆叠N层形成深度网络
        # 每个层独立处理，共享相同的配置
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # 3. 最终层归一化 - 对整个模型的输出进行归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播计算。

        参数：
        - input_ids: 输入token ID [batch_size, seq_len]
        - positions: 位置索引 [batch_size, seq_len]

        返回值：
        - hidden_states: 最终隐藏状态 [batch_size, seq_len, hidden_size]

        计算流程：
        1. 词嵌入：token ID -> 向量表示
        2. 逐层处理：每个decoder layer依次处理
        3. 最终归一化：RMSNorm归一化输出

        详细步骤：
        ```
        # 步骤1：词嵌入
        hidden_states = embed_tokens(input_ids)  # [batch, seq_len, hidden_size]
        residual = None

        # 步骤2：逐层处理
        for layer in layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 步骤3：最终归一化
        hidden_states, _ = norm(hidden_states, residual)
        return hidden_states
        ```

        残差连接：
        - residual在层间传递，避免梯度消失
        - 第一层：residual = None，保存为原始输入
        - 后续层：residual累积所有之前的输出

        KV-Cache支持：
        - positions用于RoPE位置编码
        - 在推理时，positions会包含所有历史位置
        - 支持prefill（处理新tokens）和decode（生成新tokens）

        张量并行：
        - embed_tokens在vocab维度并行
        - layers在层维度串行（每层所有GPU都执行）
        - norm在最后一个GPU执行最终归一化

        使用示例：
        ```python
        model = Qwen3Model(config)
        # 单序列推理
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 5个token
        positions = torch.tensor([[0, 1, 2, 3, 4]])
        hidden_states = model(input_ids, positions)  # [1, 5, hidden_size]

        # 批量推理
        batch_input = torch.randint(0, vocab_size, (4, 20))  # 4个序列，每个20个token
        batch_positions = torch.arange(20).unsqueeze(0).repeat(4, 1)
        batch_hidden = model(batch_input, batch_positions)  # [4, 20, hidden_size]
        ```

        性能优化：
        - 使用torch.compile加速计算
        - 支持CUDA Graph减少调度开销
        - 优化的内存访问模式（flash-attention等）
        """
        # 步骤1：词嵌入 - 将token ID转换为向量表示
        hidden_states = self.embed_tokens(input_ids)

        # 初始化残差（第一次层调用时使用）
        residual = None

        # 步骤2：逐层处理 - N个解码器层依次处理
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 步骤3：最终归一化 - RMSNorm归一化
        # 传入residual进行最终的残差连接
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3因果语言模型（带输出层）

    在Qwen3Model基础上添加输出层（LM Head），实现完整的语言建模功能：
    1. 内部包含Qwen3Model（主干网络）
    2. LM Head（语言模型头）- 从hidden states计算logits
    3. 支持词嵌入共享（weight tying）

    功能特点：
    - 因果语言建模：预测下一个token
    - 支持张量并行推理
    - 兼容prefix caching和KV-cache
    - 支持词嵌入权重共享（减少参数）

    词嵌入共享（Weight Tying）：
    - 将lm_head的权重与embed_tokens的权重绑定
    - 减少参数量（通常在1B以下模型使用）
    - 提升词嵌入质量（embed和output学习互相促进）

    使用场景：
    - 文本生成（自回归）
    - 批量推理服务
    - 聊天对话系统
    - 文本补全

    注意事项：
    - 输入的token序列需要以特殊token开始（BOS）
    - 输出是logits，需要通过softmax得到概率分布
    - 推理时需要处理KV-cache和位置索引
    """

    # 权重打包映射 - 用于从HuggingFace加载预训练权重
    # 将原始模型的模块名映射到本实现中的模块名
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),  # Q投影 -> QKV投影的Q部分
        "k_proj": ("qkv_proj", "k"),  # K投影 -> QKV投影的K部分
        "v_proj": ("qkv_proj", "v"),  # V投影 -> QKV投影的V部分
        "gate_proj": ("gate_up_proj", 0),  # 门控投影 -> GateUp投影的第0部分
        "up_proj": ("gate_up_proj", 1),  # 上投影 -> GateUp投影的第1部分
    }

    def __init__(self, config: Qwen3Config) -> None:
        """
        初始化Qwen3ForCausalLM。

        参数：
        - config: Qwen3Config，模型配置

        组件说明：
        1. model: Qwen3Model，主干Transformer网络
        2. lm_head: ParallelLMHead，输出层（从hidden states到logits）

        权重共享：
        - 如果config.tie_word_embeddings为True
        - 则lm_head.weight与model.embed_tokens.weight共享
        - 两者指向同一个张量（内存地址相同）

        优势：
        - 减少参数量（通常减少10-20%）
        - 提升小模型的性能
        - 防止过拟合

        使用示例：
        ```python
        from transformers import Qwen3Config
        config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
        model = Qwen3ForCausalLM(config)

        # 检查是否权重共享
        if config.tie_word_embeddings:
            assert model.lm_head.weight is model.model.embed_tokens.weight
        ```
        """
        super().__init__()

        # 1. 主干网络 - Qwen3Model
        self.model = Qwen3Model(config)

        # 2. 语言模型头 - 从hidden states计算logits
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        # 3. 词嵌入权重共享（可选）
        if config.tie_word_embeddings:
            # 将lm_head的权重指向embed_tokens的权重
            # 注意：这是同一块内存的引用，不是copy
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播 - 计算主干网络的隐藏状态。

        参数：
        - input_ids: 输入token ID [batch_size, seq_len]
        - positions: 位置索引 [batch_size, seq_len]

        返回值：
        - hidden_states: 主干网络输出的隐藏状态 [batch_size, seq_len, hidden_size]

        说明：
        - 这个方法直接调用内部model的forward
        - 不包含最终的logits计算
        - 适用于需要中间表示的场景

        使用场景：
        - 需要提取hidden states用于其他任务（如特征提取）
        - 在nano-vllm中，内部推理使用此方法
        - 计算logits需要额外调用compute_logits

        使用示例：
        ```python
        model = Qwen3ForCausalLM(config)
        input_ids = torch.randint(0, vocab_size, (2, 10))
        positions = torch.arange(10).unsqueeze(0).repeat(2, 1)
        hidden_states = model(input_ids, positions)  # [2, 10, hidden_size]
        ```
        """
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算最终的logits（概率分布的对数几率）。

        参数：
        - hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]

        返回值：
        - logits: 对数几率 [batch_size, seq_len, vocab_size]

        说明：
        - 将hidden states通过线性层映射到vocab_size维度
        - 不包含softmax（避免数值不稳定）
        - 在sampler中会应用温度和采样策略

        计算流程：
        ```
        logits = hidden_states @ lm_head.weight^T + lm_head.bias
        ```

        内存布局：
        - 输入：[batch_size, seq_len, hidden_size]
        - 输出：[batch_size, seq_len, vocab_size]

        张量并行：
        - 在prefill阶段，每个GPU计算部分logits，然后聚合
        - 在decode阶段，只计算最后一个token的logits

        使用示例：
        ```python
        model = Qwen3ForCausalLM(config)
        hidden_states = model(input_ids, positions)  # [2, 10, hidden_size]
        logits = model.compute_logits(hidden_states)  # [2, 10, vocab_size]

        # 转换为概率分布
        probs = torch.softmax(logits, dim=-1)

        # 获取每个位置的top token
        top_tokens = torch.argmax(probs, dim=-1)
        ```

        性能优化：
        - 使用torch.compile加速矩阵乘法
        - 优化的线性层实现（ParallelLMHead）
        - 支持CUDA Graph减少调度开销
        """
        return self.lm_head(hidden_states)
