from dataclasses import dataclass


@dataclass
class SamplingParams:
    """采样参数，控制 temperature、最大生成长度及是否忽略 EOS 等生成行为。"""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
