from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV-cache块类。

    核心功能：
    1. 管理单个KV-cache块的状态
    2. 跟踪引用计数（ref_count）
    3. 存储块的哈希值（用于prefix caching）
    4. 存储块内的token数据

    引用计数机制：
    - ref_count = 0: 块是空闲的，可被分配
    - ref_count > 0: 块被使用中，不能释放
    - 当ref_count减至0时，块回到空闲池

    Prefix Caching：
    - 通过哈希值识别相同内容的块
    - 相同哈希的块可以共享，避免重复计算
    - 使用xxhash进行高效哈希计算
    """

    def __init__(self, block_id):
        """
        初始化块。

        使用方法：
        block = Block(block_id)

        参数：
        - block_id: 全局唯一块ID

        初始状态：
        - 空闲状态（ref_count = 0）
        - 无效哈希（hash = -1）
        - 无token数据
        """
        self.block_id = block_id
        self.ref_count = 0  # 引用计数，0表示空闲
        self.hash = -1  # 块的哈希值，-1表示无效
        self.token_ids = []  # 块内的token列表

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的哈希和token数据。

        使用方法：
        block.update(hash_value, token_list)

        参数：
        - hash: 块的哈希值（用于prefix caching）
        - token_ids: 块内的token列表

        使用场景：
        1. 块被填满后更新哈希
        2. 建立prefix caching索引
        3. 标记块的内容
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块状态（分配时调用）。

        使用方法：
        block.reset()

        重置内容：
        - ref_count设为1（表示被使用）
        - hash重置为-1（无效）
        - token_ids清空

        使用场景：
        1. 从空闲池分配块时
        2. 重新使用一个旧块时
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    块管理器，负责管理所有KV-cache块。

    核心功能：
    1. 分配和释放KV-cache块
    2. 维护块的引用计数
    3. 实现prefix caching
    4. 管理空闲块池

    架构设计：
    - 使用哈希表（hash_to_block_id）实现快速查找
    - 使用双端队列（free_block_ids）管理空闲块
    - 使用集合（used_block_ids）跟踪已使用块

    Prefix Caching策略：
    1. 计算每个block的哈希值
    2. 将哈希映射到block ID
    3. 新序列分配时检查是否有相同哈希的块
    4. 共享相同内容的块，避免重复计算
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器。

        使用方法：
        manager = BlockManager(num_blocks=1000, block_size=256)

        参数：
        - num_blocks: 总块数
        - block_size: 每块的token数（默认256）

        初始化内容：
        1. 创建所有块实例
        2. 初始化哈希映射表
        3. 初始化空闲块队列（包含所有块）
        4. 初始化已使用块集合（初始为空）
        """
        self.block_size = block_size

        # 创建所有块
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # 哈希到block ID的映射（用于prefix caching）
        self.hash_to_block_id: dict[int, int] = dict()

        # 空闲块ID的双端队列（高效pop from left）
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # 已使用块的集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算token序列的哈希值。

        使用方法：
        hash_value = BlockManager.compute_hash(token_list, prefix_hash)

        参数：
        - token_ids: 要哈希的token列表
        - prefix: 前一个块的哈希值（用于链式哈希）

        返回值：
        - 64位哈希值

        用途：
        1. 识别相同内容的块
        2. 实现prefix caching
        3. 支持块级共享

        哈希算法：
        - 使用xxhash（快速非加密哈希）
        - 支持链式哈希（累积式计算）
        """
        h = xxhash.xxh64()

        # 如果有前缀，先添加前缀哈希
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))

        # 添加token数据
        h.update(np.array(token_ids).tobytes())

        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配指定ID的块（内部方法）。

        使用方法：
        内部使用，由allocate()调用
        block = self._allocate_block(block_id)

        参数：
        - block_id: 要分配的块ID

        返回值：
        - 分配到的块对象

        操作：
        1. 获取块对象
        2. 检查块是否空闲（ref_count == 0）
        3. 重置块状态
        4. 从空闲队列移除
        5. 加入已使用集合
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放指定ID的块（内部方法）。

        使用方法：
        内部使用，由deallocate()调用
        block = self._deallocate_block(block_id)

        参数：
        - block_id: 要释放的块ID

        操作：
        1. 检查块引用计数是否为0
        2. 从已使用集合移除
        3. 加入空闲队列
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否可以为一个序列分配块。

        使用方法：
        if manager.can_allocate(seq):
            manager.allocate(seq)

        参数：
        - seq: 要分配块的序列

        返回值：
        - True: 有足够的空闲块
        - False: 空闲块不足

        判断条件：
        - 空闲块数 >= 序列需要的块数
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配KV-cache块。

        使用方法：
        manager.allocate(seq)

        参数：
        - seq: 需要分配块的序列

        分配策略：
        1. 顺序为序列的每个block分配空间
        2. 优先使用相同内容的缓存块（prefix caching）
        3. 如果缓存命中，增加引用计数
        4. 如果缓存未命中，从空闲池分配新块
        5. 为新块计算哈希并建立索引

        使用场景：
        1. 序列开始prefill时
        2. 序列需要更多KV-cache空间时
        3. 抢占后重新分配时

        注意：
        - 支持prefix caching（相同内容共享块）
        - 块级优化，而非token级
        - 最后一个块可能不满（不满一个block_size）
        """
        assert not seq.block_table

        h = -1
        cache_miss = False

        # 遍历序列的每个block
        for i in range(seq.num_blocks):
            # 获取该block的token
            token_ids = seq.block(i)

            # 计算哈希（仅对满的block计算）
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )

            # 查找缓存的块
            block_id = self.hash_to_block_id.get(h, -1)

            # 检查是否真正命中缓存（哈希冲突检查）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 缓存未命中

            if cache_miss:
                # 缓存未命中：从空闲池分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：可能共享已有块
                seq.num_cached_tokens += self.block_size  # 增加缓存token计数

                if block_id in self.used_block_ids:
                    # 块正在使用中，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块是空闲的，分配并初始化
                    block = self._allocate_block(block_id)

            # 更新块的哈希和索引
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 记录到序列的block table
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放序列占用的所有块。

        使用方法：
        manager.deallocate(seq)

        参数：
        - seq: 要释放块的序列

        释放策略：
        1. 逆序遍历序列的block table
        2. 减少每个块的引用计数
        3. 当引用计数降至0时，释放块回空闲池
        4. 清空序列的缓存信息

        使用场景：
        1. 序列完成时
        2. 序列被抢占时
        3. 序列结束时资源清理

        注意：
        - 逆序释放（通常最后分配的先释放）
        - 只有当ref_count降至0时才真正释放
        """
        # 逆序遍历block table
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少引用计数

            # 引用计数为0时可以释放
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        # 清空序列的缓存信息
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以向序列追加新token。

        使用方法：
        if manager.can_append(seq):
            manager.may_append(seq)

        参数：
        - seq: 要检查的序列

        返回值：
        - True: 可以追加（有空闲块或当前块未满）
        - False: 无法追加（需要新块但没有空闲块）

        判断逻辑：
        - 如果当前block未满（len(seq) % block_size != 1），可以直接追加
        - 如果当前block刚满（len(seq) % block_size == 1），需要新块
        - 需要新块时，检查是否有空闲块
        """
        # 需要新块的条件：当前长度 % block_size == 1
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        尝试向序列追加新token（可能需要分配新块）。

        使用方法：
        manager.may_append(seq)

        参数：
        - seq: 要追加的序列

        处理逻辑：
        1. 如果当前block刚满（需要新块）：
           - 从空闲池分配新块
           - 添加到block table

        2. 如果当前block刚填满（变为满块）：
           - 计算块哈希
           - 更新块的哈希和索引
           - 支持后续的prefix caching

        3. 其他情况（块未满）：
           - 无需操作

        注意：
        - 必须在append_token()之前调用
        - 确保有足够的空间存储新token
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        # 情况1：当前block刚满，需要新块
        # 例如：block_size=256，当前长度=256*k + 1
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1

            # 分配新块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        # 情况2：当前block刚填满，可以计算哈希
        # 例如：block_size=256，当前长度=256*k
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1

            # 获取刚填满的block的token
            token_ids = seq.block(seq.num_blocks - 1)

            # 计算前缀哈希
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1

            # 计算当前block的哈希
            h = self.compute_hash(token_ids, prefix)

            # 更新块的哈希和索引
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        # 情况3：当前block未满，无需操作
        else:
            assert last_block.hash == -1
