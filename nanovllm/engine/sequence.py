from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举。

    状态流转：
    WAITING -> RUNNING -> FINISHED

    WAITING：
    - 初始状态，序列已提交但未开始处理
    - 位于调度器的waiting队列中
    - 等待被调度器选中

    RUNNING：
    - 序列正在被处理
    - 位于调度器的running队列中
    - 占用KV-cache资源

    FINISHED：
    - 序列已完成生成
    - 已释放所有资源
    - 等待收集结果
    """

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    序列类，表示一个完整的文本生成请求。

    核心功能：
    1. 存储输入prompt和生成的completion
    2. 跟踪序列状态和采样参数
    3. 管理KV-cache块分配信息
    4. 支持序列的序列化和反序列化

    使用场景：
    1. 每次generate()调用会创建多个Sequence对象
    2. 贯穿整个推理生命周期
    3. 用于追踪单个请求的状态

    生命周期：
    1. 创建：add_request()时创建Sequence
    2. 等待：进入调度器waiting队列
    3. 运行：被调度器选中，开始prefill/decode
    4. 完成：达到结束条件，状态变为FINISHED
    """

    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        """
        初始化序列。

        使用方法：
        seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=0.7))

        参数：
        - token_ids: 输入的token ID列表
        - sampling_params: 采样参数

        初始化内容：
        1. 分配全局唯一序列ID
        2. 设置初始状态为WAITING
        3. 复制token列表（避免外部修改）
        4. 记录最后一个token
        5. 统计token数量
        6. 设置采样参数
        7. 初始化缓存和块表

        注意：
        - token_ids会被复制，避免外部修改影响内部状态
        - num_cached_tokens初始为0，表示没有缓存
        - block_table为空，等待分配KV-cache块
        """
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
        """
        获取序列长度。

        使用方法：
        length = len(seq)

        返回值：
        - 序列的总token数（prompt + completion）
        """
        return self.num_tokens

    def __getitem__(self, key):
        """
        获取指定位置的token。

        使用方法：
        token = seq[index]

        参数：
        - key: token索引

        返回值：
        - 指定位置的token ID

        注意：
        支持切片操作，如seq[5:10]获取子序列
        """
        return self.token_ids[key]

    @property
    def is_finished(self):
        """
        检查序列是否已完成。

        使用方法：
        if seq.is_finished:
            print("序列已完成")

        返回值：
        - True: 状态为FINISHED
        - False: 状态为WAITING或RUNNING
        """
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """
        获取已生成的completion token数。

        使用方法：
        num_completion = seq.num_completion_tokens

        返回值：
        - completion的token数（总token - prompt token）
        """
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """
        获取prompt的token列表。

        使用方法：
        prompt_tokens = seq.prompt_token_ids

        返回值：
        - prompt的token ID列表（不变）
        """
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """
        获取completion的token列表。

        使用方法：
        completion_tokens = seq.completion_token_ids

        返回值：
        - 已生成的completion token ID列表
        """
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_cached_blocks(self):
        """
        获取已缓存的block数量。

        使用方法：
        num_blocks = seq.num_cached_blocks

        返回值：
        - 已缓存的block数（用于prefix caching）
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        获取总block数量。

        使用方法：
        total_blocks = seq.num_blocks

        返回值：
        - 序列占用的总block数
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        获取最后一个block的有效token数。

        使用方法：
        last_block_size = seq.last_block_num_tokens

        返回值：
        - 最后一个block中的有效token数（可能小于block_size）
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取指定block的token。

        使用方法：
        tokens_in_block = seq.block(3)

        参数：
        - i: block索引

        返回值：
        - 该block中的所有token

        注意：
        最后一个block可能不满（返回实际token数）
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """
        添加一个新生成的token。

        使用方法：
        seq.append_token(new_token_id)

        参数：
        - token_id: 新生成的token ID

        更新内容：
        1. 将token添加到token_ids列表
        2. 更新last_token
        3. 增加总token数

        使用场景：
        1. decode阶段每次生成新token后调用
        2. 构建完整的输出序列
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化时保存的状态。

        使用方法：
        内部使用，pickle模块自动调用

        优化策略：
        - 如果没有completion（num_completion_tokens == 0），只保存完整的token_ids
        - 如果有completion，只保存last_token，节省空间
        - 其他元数据（num_tokens等）总是保存

        注意：
        这是为了减少多进程通信的数据量
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            # 优化：如果没有completion，保存完整token_ids；否则只保存last_token
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        """
        反序列化时恢复的状态。

        使用方法：
        内部使用，pickle模块自动调用

        恢复策略：
        1. 先恢复元数据（num_tokens等）
        2. 根据是否有completion决定恢复方式
        3. 如果有completion，token_ids需要从prompt重建

        注意：
        - 需要从其他序列同步seq_id和status
        - 这是因为pickle不保存动态生成的值
        """
        # 恢复元数据
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]

        # 恢复token数据
        if self.num_completion_tokens == 0:
            # 没有completion，完整token_ids直接可用
            self.token_ids = state[-1]
        else:
            # 有completion，只保存了last_token
            self.last_token = state[-1]
