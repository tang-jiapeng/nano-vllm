from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    全局上下文，用于在推理过程中传递信息。

    主要用途：
    - 在模型的不同组件之间传递上下文信息
    - 避免在函数参数中传递大量张量
    - 支持flash-attn的varlen和blocked模式

    字段说明：
    - is_prefill: 是否为prefill阶段
    - cu_seqlens_q/k: 累积序列长度（用于varlen attention）
    - max_seqlen_q/k: 最大序列长度
    - slot_mapping: KV-cache槽位映射
    - context_lens: 每个序列的上下文长度
    - block_tables: 块表（用于blocked KV-cache）
    """

    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


_CONTEXT = Context()


def get_context():
    """
    获取当前全局上下文。

    使用方法：
    context = get_context()

    返回值：
    - 全局Context对象

    使用场景：
    1. 在模型组件中获取当前推理的上下文信息
    2. flash-attn需要这些信息进行变长序列处理
    """
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    """
    设置全局上下文。

    使用方法：
    set_context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        slot_mapping=slot_mapping
    )

    参数：
    - is_prefill: 是否为prefill阶段
    - 其他参数：见Context类字段说明

    使用场景：
    1. ModelRunner在推理前设置上下文
    2. 为模型组件提供必要的上下文信息
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context():
    """
    重置全局上下文为空。

    使用方法：
    reset_context()

    使用场景：
    1. 推理完成后清理上下文
    2. 避免上下文污染下一次推理
    """
    global _CONTEXT
    _CONTEXT = Context()
