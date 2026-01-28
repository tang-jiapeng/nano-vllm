import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    推理配置类，包含所有运行时参数。

    核心参数说明：

    模型相关：
    - model: 模型路径（必须为本地目录）

    性能调优：
    - max_num_batched_tokens: 批处理最大token数（默认16384）
    - max_num_seqs: 最大并发序列数（默认512）
    - max_model_len: 最大序列长度（默认4096）
    - gpu_memory_utilization: GPU显存利用率（默认0.9）

    并行配置：
    - tensor_parallel_size: 张量并行大小（默认1）

    执行模式：
    - enforce_eager: 是否强制使用eager模式（默认False，即启用CUDA Graph）

    KV-cache：
    - kvcache_block_size: KV-cache块大小（默认256，必须是256的倍数）
    - num_kvcache_blocks: KV-cache块数量（初始化时自动计算）

    使用方法：
    config = Config(
        model="/path/to/model",
        max_num_seqs=256,
        tensor_parallel_size=2
    )
    """

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
