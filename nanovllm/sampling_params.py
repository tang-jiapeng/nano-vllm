from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    采样参数，控制文本生成的行为。

    核心参数：

    温度控制：
    - temperature: 采样温度（默认1.0）
      * 值越大，输出越随机
      * 值越小，输出越确定性
      * 必须 > 1e-10（不支持greedy sampling）

    生成限制：
    - max_tokens: 最大生成token数（默认64）
    - ignore_eos: 是否忽略EOS token（默认False）

    使用方法：
    # 基本采样
    params = SamplingParams(temperature=0.7, max_tokens=100)

    # 忽略EOS（用于代码生成等场景）
    params = SamplingParams(temperature=0.5, max_tokens=512, ignore_eos=True)
    """

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
