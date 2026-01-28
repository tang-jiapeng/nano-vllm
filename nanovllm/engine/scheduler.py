from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    调度器，负责管理请求的调度和状态转换。

    核心功能：
    1. 管理两个队列：waiting（等待队列）和running（运行队列）
    2. 实现两级调度：prefill（预填入）和decode（解码）
    3. 批处理优化，平衡吞吐量和延迟
    4. 管理KV-cache块的分配和释放
    5. 处理内存不足时的抢占（preemption）

    调度策略：
    - Prefill阶段：批量处理所有新到达的请求，优先高吞吐量
    - Decode阶段：逐个生成token，优先低延迟
    - 内存管理：当KV-cache不足时，优先抢占最长的序列
    """

    def __init__(self, config: Config):
        """
        初始化调度器

        使用方法：
        scheduler = Scheduler(config)

        参数：
        - config: 配置对象，包含所有调度相关的参数

        初始化内容：
        1. 设置最大并发序列数和批处理token数
        2. 创建BlockManager管理KV-cache
        3. 初始化waiting和running队列
        """
        # 最大并发序列数（同时运行的序列数量）
        self.max_num_seqs = config.max_num_seqs

        # 批处理的最大token数（一次处理的token总数）
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # 结束符ID，用于判断序列是否结束
        self.eos = config.eos

        # 创建KV-cache块管理器
        # 负责分配、释放和管理KV-cache块，支持prefix caching
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )

        # 等待队列：包含所有已提交但未开始处理的序列
        self.waiting: deque[Sequence] = deque()

        # 运行队列：包含所有正在处理的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """
        检查是否所有序列都已完成。

        使用方法：
        if scheduler.is_finished():
            print("所有请求已完成")

        返回值：
        - True: 所有队列都为空，调度完成
        - False: 还有未完成的序列

        使用场景：
        1. 判断是否继续推理循环
        2. 监控调度器状态
        3. 优雅退出判断条件
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加一个新序列到等待队列

        使用方法：
        scheduler.add(seq)

        参数：
        - seq: Sequence对象，包含token ID和采样参数

        使用场景：
        1. 接收新的推理请求
        2. 将抢占的序列重新加入等待队列
        3. 请求重试时重新排队

        处理流程：
        直接将序列追加到waiting队列的尾部，FIFO顺序处理
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        执行一次调度决策，选择要处理的序列批次。

        使用方法：
        通常由LLMEngine.step()调用，无需手动调用
        seqs, is_prefill = scheduler.schedule()

        返回值：
        - scheduled_seqs: 选中的序列列表
        - is_prefill: 是否为prefill阶段
          * True: prefill阶段，批量处理输入token
          * False: decode阶段，逐个生成新token

        调度策略：

        1. Prefill阶段（优先处理waiting队列）：
           - 顺序处理waiting队列中的序列
           - 直到达到最大序列数或最大token数限制
           - 检查KV-cache是否足够分配
           - 分配block并更新状态为RUNNING

        2. Decode阶段（处理running队列）：
           - 如果waiting为空，处理running中的序列
           - 检查是否可以追加新token到KV-cache
           - 内存不足时执行抢占（preemption）
           - 将序列重新放回队列（preserve顺序）

        使用场景：
        1. 批量处理新到达的请求（prefill）
        2. 继续生成进行中的请求（decode）
        3. 内存回收和负载均衡
        """
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # ===== Prefill阶段：处理等待队列 =====
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # 检查是否超出批处理token数限制
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                break

            # 检查KV-cache块是否足够分配
            if not self.block_manager.can_allocate(seq):
                break

            # 选择该序列进行处理
            num_seqs += 1
            self.block_manager.allocate(seq)  # 分配KV-cache块

            # 统计新处理的token数（排除缓存的token）
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            # 更新序列状态并移动队列
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()  # 从等待队列移除
            self.running.append(seq)  # 加入运行队列
            scheduled_seqs.append(seq)

        # 如果有prefill序列，直接返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # ===== Decode阶段：处理运行队列 =====
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # 检查是否可以追加新token（KV-cache空间检查）
            while not self.block_manager.can_append(seq):
                # 内存不足，执行抢占
                if self.running:
                    # 抢占队列中最长的序列
                    self.preempt(self.running.pop())
                else:
                    # 只能抢占当前序列
                    self.preempt(seq)
                    break  # 跳出内层while循环
            else:
                # 内存充足，可以处理
                num_seqs += 1
                self.block_manager.may_append(seq)  # 可能需要分配新块
                scheduled_seqs.append(seq)

        # 保证队列顺序：将decode的序列放回队列前端（reverse twice preserves order）
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占一个序列，释放其占用的资源。

        使用方法：
        通常在内存不足时自动调用，无需手动调用
        scheduler.preempt(seq)

        参数：
        - seq: 要抢占的序列

        抢占策略：
        1. 优先抢占最长的序列（释放更多KV-cache）
        2. 将序列状态从RUNNING改为WAITING
        3. 释放所有KV-cache块
        4. 重新加入等待队列头部（高优先级重新调度）

        使用场景：
        1. KV-cache空间不足，无法处理新token
        2. 内存压力过大，需要回收资源
        3. 负载均衡，避免某个序列占用过多资源

        注意：
        抢占会导致序列重新处理（recompute），但保证了系统的稳定运行
        """
        # 1. 更新序列状态
        seq.status = SequenceStatus.WAITING

        # 2. 释放KV-cache块
        self.block_manager.deallocate(seq)

        # 3. 重新加入等待队列头部（优先重新调度）
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        后处理序列，更新状态和资源。

        使用方法：
        通常在模型推理完成后由LLMEngine.step()调用
        finished_flags = scheduler.postprocess(seqs, token_ids)

        参数：
        - seqs: 刚处理完的序列列表
        - token_ids: 生成的token ID列表

        返回值：
        - finished_flags: 每个序列是否完成的布尔列表

        处理流程：
        1. 将生成的token添加到序列中
        2. 检查序列是否应该结束：
           - 遇到EOS token且未设置ignore_eos
           - 达到最大生成token数
        3. 如果序列完成：
           - 更新状态为FINISHED
           - 释放KV-cache块
           - 从运行队列移除
        4. 否则继续留在运行队列中

        使用场景：
        1. 每次模型推理后的状态更新
        2. 完成序列的清理工作
        3. 资源回收和复用

        注意：
        只有rank 0（主进程）会调用此方法，因为只有它有采样结果
        """
        finished_flags = []

        for seq, token_id in zip(seqs, token_ids):
            # 1. 将生成的token添加到序列中
            seq.append_token(token_id)

            # 2. 检查是否应该结束
            should_finish = False

            # 情况1：遇到EOS token且未忽略EOS
            if not seq.ignore_eos and token_id == self.eos:
                should_finish = True

            # 情况2：达到最大生成token数
            elif seq.num_completion_tokens == seq.max_tokens:
                should_finish = True

            # 3. 如果应该结束，更新状态并释放资源
            if should_finish:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放KV-cache
                self.running.remove(seq)  # 从运行队列移除

            finished_flags.append(should_finish)

        return finished_flags
