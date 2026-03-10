import logging
import os
from dataclasses import dataclass

from transformers import AutoConfig

from nanovllm.layers.quantization.quant_method import AWQConfig

try:
    import flash_attn as _  # noqa: F401

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """推理配置，包含模型路径、调度参数、tensor parallel 及 KV-cache 等运行时选项。"""

    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    chunked_prefill: bool = False
    hf_config: AutoConfig | None = None
    awq_config: AWQConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if HAS_FLASH_ATTN:
            assert (
                self.kvcache_block_size % 256 == 0
            ), "flash-attn 要求 kvcache_block_size 为 256 的倍数"
        else:
            assert (
                self.kvcache_block_size >= 16
                and (self.kvcache_block_size & (self.kvcache_block_size - 1)) == 0
            ), "kvcache_block_size 必须为 ≥16 的 2 的幂"
            logger.warning(
                "flash-attn 未安装，将使用 Triton 自实现 attention kernel。"
                "安装 flash-attn 以获得最佳性能: pip install flash-attn"
            )
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len

        # MoE 已使用 fused_moe 实现，默认允许 CUDA Graph。

        # 自动检测 AWQ 量化配置
        self.awq_config = AWQConfig.from_json(self.model)
        if self.awq_config is not None:
            logger.info(
                f"检测到 AWQ 量化模型（weight_bits={self.awq_config.weight_bits}, "
                f"group_size={self.awq_config.group_size}）。"
            )
