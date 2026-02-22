import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型运行器，负责模型加载、KV-cache 分配、CUDA Graph 捕获及多进程通信。
    rank 0 主进程负责任务分发和采样，rank > 0 子进程仅负责前向计算。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化 ModelRunner。

        流程：初始化 NCCL 分布式环境 -> 加载模型 -> warmup ->
              分配 KV-cache -> 捕获 CUDA Graph -> 初始化 SharedMemory。
        rank > 0 的子进程初始化后直接进入 loop() 等待指令。
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 1. 初始化 NCCL 分布式环境
        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )

        # 2. 设置 CUDA 设备
        torch.cuda.set_device(rank)

        # 3. 加载模型
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # 4. Warmup
        self.warmup_model()

        # 5. 分配 KV-cache
        self.allocate_kv_cache()

        # 6. 捕获 CUDA Graph
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 7. 初始化进程间 SharedMemory 通信
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """清理资源：关闭 SharedMemory、删除 CUDA Graph、销毁分布式进程组。"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()

        if not self.enforce_eager:
            del self.graphs, self.graph_pool

        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """子进程消息循环：持续从 SharedMemory 读取并执行指令，直到收到 exit。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """子进程从 SharedMemory 读取指令（通过 Event 等待唤醒）。协议: [4B 长度][pickle 数据]。"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """主进程向 SharedMemory 写入指令并通过 Event 唤醒所有子进程。"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """统一调用接口：rank 0 先写 SharedMemory 再本地执行，rank > 0 直接执行。"""
        # 多进程模式：主进程先通过 SharedMemory 分发指令
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)

        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """模型预热：用虚拟序列执行一次 prefill，触发 CUDA kernel JIT 编译和内存分配器初始化。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )

        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        根据剩余显存分配 KV-cache。

        布局: [2(K/V), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        各 attention 层共享同一 KV-cache 实例。
        """
        config = self.config
        hf_config = config.hf_config

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )

        # 单个 block 的字节数（K + V）
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.torch_dtype.itemsize
        )

        # 根据显存利用率计算可分配 block 数
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        # 将 KV-cache 绑定到各 attention 层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """构建 [num_seqs, max_num_blocks] 的 block table tensor，无效位填 -1。"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 prefill 阶段数据：拼接各序列新 token、计算 cu_seqlens 和 slot_mapping，
        设置全局 context。跳过已被 prefix caching 命中的 token。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue

            # 为新分配的 block 建立 slot mapping
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        # 存在 prefix cache 时需要 block tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 转为 tensor 并传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 decode 阶段数据：每个序列取最后一个 token，
        计算 position、context_len、slot_mapping 和 block_tables。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))

            # 当前 token 在最后一个 block 中的 slot 位置
            last_block_idx = seq.block_table[-1]
            slot = last_block_idx * self.block_size + seq.last_block_num_tokens - 1
            slot_mapping.append(slot)

        # 转为 tensor 并传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)

        block_tables = self.prepare_block_tables(seqs)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """提取各序列的 temperature 参数，返回 GPU tensor。仅 rank 0 调用。"""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)

        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        """
        执行模型前向推理。

        prefill 或 batch > 512 时使用 eager 模式；
        decode 阶段使用预捕获的 CUDA Graph 加速。
        """
        # Prefill / 强制 eager / 大 batch 时直接执行
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph 模式
            bs = input_ids.size(0)
            context = get_context()

            # 选择第一个 >= bs 的预捕获 graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # 更新预分配变量中的有效数据
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions

            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping

            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens

            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)
            ] = context.block_tables

            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        完整推理流程：数据准备 -> 前向计算 -> 采样 -> 清理。
        rank 0 返回 token_ids，rank > 0 返回 None。
        """
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)

        # 仅主进程执行采样
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )

        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        为多个预定义 batch size 捕获 CUDA Graph，加速 decode 推理。
        逆序捕获（大 batch 优先），所有 graph 共享同一 memory pool。
        """
        config = self.config
        hf_config = config.hf_config

        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 预分配所有 graph 共用的输入/输出变量
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 预定义 batch sizes: [1, 2, 4, 8, 16, 32, ...]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 逆序逐个捕获 CUDA Graph
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()

            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            # Warmup：触发 CUDA kernel JIT 编译
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # 捕获计算图
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # 首个 graph 创建后获取共享内存池
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存预分配变量引用，供 run_model() 使用
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
