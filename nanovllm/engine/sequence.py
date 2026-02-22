from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列状态枚举。流转: WAITING -> RUNNING -> FINISHED。"""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """表示单个文本生成请求，贯穿整个推理生命周期，管理 token、状态及 KV-cache block 信息。"""

    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        """初始化序列。token_ids 会被复制以避免外部修改。"""
        # 全局唯一序列ID
        self.seq_id = next(Sequence.counter)

        # 初始状态
        self.status = SequenceStatus.WAITING

        # 复制token列表
        self.token_ids = copy(token_ids)

        # 当前最后一个token（用于decode阶段）
        self.last_token = token_ids[-1]

        # 总token数（prompt + completion）
        self.num_tokens = len(self.token_ids)

        # prompt的token数（不变）
        self.num_prompt_tokens = len(token_ids)

        # 已缓存的token数（prefix caching）
        self.num_cached_tokens = 0

        # KV-cache块表，记录该序列使用的所有块
        self.block_table = []

        # 采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回序列总 token 数（prompt + completion）。"""
        return self.num_tokens

    def __getitem__(self, key):
        """按索引或切片获取 token ID。"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """序列是否已完成生成。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 completion token 数。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取 prompt 部分的 token ID 列表。"""
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取已生成的 completion token ID 列表。"""
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        """已被 prefix caching 命中的 block 数。"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """序列占用的总 block 数（向上取整）。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个 block 中的有效 token 数（可能不满）。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """返回第 i 个 block 对应的 token 子列表。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """追加新生成的 token 并更新内部计数。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """pickle 序列化：无 completion 时保存完整 token_ids，否则仅保存 last_token 以减少通信量。"""
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            # 优化：如果没有completion，保存完整token_ids；否则只保存last_token
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        """pickle 反序列化：恢复元数据及 token 信息，seq_id 和 status 需外部同步。"""
        # 恢复元数据
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]

        # 恢复token数据
        if self.num_completion_tokens == 0:
            # 没有completion，完整token_ids直接可用
            self.token_ids = state[-1]
        else:
            # 有completion，只保存了last_token
            self.last_token = state[-1]
