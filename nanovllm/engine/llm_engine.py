import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    推理引擎核心类，协调 Scheduler、ModelRunner 和 Tokenizer，
    支持 tensor parallel 多进程推理。
    """

    def __init__(self, model, **kwargs):
        """
        初始化推理引擎。

        Args:
            model: 本地模型目录路径
            **kwargs: 传递给 Config 的参数

        流程：解析配置 -> 启动 tensor parallel 子进程 -> 创建 ModelRunner ->
              加载 Tokenizer -> 创建 Scheduler -> 注册 atexit 清理
        """
        # 提取 Config 相关参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.ps = []
        self.events = []

        ctx = mp.get_context("spawn")

        # 为 rank 1..N-1 启动子进程
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # rank 0 在主进程运行
        self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = Scheduler(config)

        atexit.register(self.exit)

    def exit(self):
        """发送 exit 指令给所有 ModelRunner，等待子进程结束，释放资源。"""
        # 向主进程的 ModelRunner 发送退出指令
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """将 prompt（字符串或 token ID 列表）封装为 Sequence 并加入调度器。"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一次推理步骤。

        使用方法：
        通常在generate()内部循环调用，不需要手动调用
        output, num_tokens = llm.step()

        使用场景：
        1. generate()中的主循环，每次迭代调用一次
        2. 手动控制推理过程时使用
        3. 调试时观察每个step的行为

        返回值：
        - outputs: 已完成序列的列表，每个元素为(seq_id, completion_token_ids)
        - num_tokens: 本次处理的token数量
          * 正数：表示prefill阶段处理的token数
          * 负数：表示decode阶段的序列数（用于计算decode吞吐量）

        执行流程：
        1. 调度器选择待处理的序列批次
        2. 调用ModelRunner执行模型推理
        3. 调度器后处理，更新序列状态
        4. 返回完成的序列和统计信息
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        return outputs, num_tokens

    def is_finished(self):
        """检查所有请求是否已完成。"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量文本生成主接口。

        Args:
            prompts: 字符串列表或 token ID 列表
            sampling_params: 单个或与 prompts 等长的采样参数
            use_tqdm: 是否显示进度条

        Returns:
            每个元素包含 {"text": str, "token_ids": list[int]}
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 按 seq_id 排序并 decode
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]

        if use_tqdm:
            pbar.close()

        return outputs
