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
    LLM推理引擎的核心类，负责整个推理流程的协调和管理。

    架构设计：
    - 采用生产者-消费者模式，Scheduler作为生产者生成待处理的序列批次
    - ModelRunner作为消费者执行模型推理
    - 支持张量并行，可将模型分布在多个GPU上

    主要功能：
    1. 初始化模型、调度器、分词器等组件
    2. 管理多进程环境下的张量并行
    3. 提供generate接口进行批量推理
    4. 处理请求的生命周期管理
    """

    def __init__(self, model, **kwargs):
        """
        初始化LLM推理引擎。

        使用方法：
        llm = LLMEngine("/path/to/model", tensor_parallel_size=2, max_num_seqs=256)

        参数：
        - model: 模型路径（必须为本地目录）
        - **kwargs: 配置参数，会与Config类的字段匹配

        初始化流程：
        1. 解析配置参数，创建Config对象
        2. 如果启用张量并行（tensor_parallel_size > 1），创建多个子进程
           - rank 0: 主进程，负责调度和结果聚合
           - rank > 0: 子进程，负责模型分片的推理
        3. 初始化ModelRunner（模型运行器）
        4. 加载分词器
        5. 创建Scheduler（调度器）
        6. 注册退出清理函数

        使用场景：
        - 单GPU推理：tensor_parallel_size=1，使用单个进程
        - 多GPU推理：tensor_parallel_size=N，创建N-1个子进程进行张量并行
        """
        # 从kwargs中提取Config相关的参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 初始化多进程相关变量
        self.ps = []  # 存储所有子进程
        self.events = []  # 存储进程间通信事件

        # 创建多进程上下文，使用spawn模式（Windows兼容性更好）
        ctx = mp.get_context("spawn")

        # 创建子进程进行张量并行（从rank 1开始，rank 0在主进程）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 创建事件用于进程间同步
            # 启动子进程，每个子进程运行一个ModelRunner实例
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 在主进程（rank 0）创建ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载分词器，use_fast=True使用快速分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id  # 设置结束符ID

        # 创建调度器，管理请求的调度和状态
        self.scheduler = Scheduler(config)

        # 注册退出时的清理函数，确保进程正常退出
        atexit.register(self.exit)

    def exit(self):
        """
        退出时清理资源。

        使用方法：
        - 通常通过atexit自动调用，无需手动调用
        - 在需要主动退出时调用：llm.exit()

        使用场景：
        1. 程序正常退出时的资源清理
        2. 进程池的优雅关闭
        3. 释放CUDA显存和进程组

        执行流程：
        1. 向所有ModelRunner发送exit指令
        2. 删除主进程的ModelRunner实例
        3. 等待所有子进程结束
        """
        # 向主进程的ModelRunner发送退出指令
        self.model_runner.call("exit")

        # 删除ModelRunner实例，触发__del__进行清理
        del self.model_runner

        # 等待所有子进程结束
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个推理请求到调度器。

        使用方法：
        llm.add_request("你好，世界", SamplingParams(temperature=0.7, max_tokens=100))

        参数：
        - prompt: 输入提示，可以是字符串或已编码的token ID列表
        - sampling_params: 采样参数，控制生成行为

        使用场景：
        1. 添加单个文本生成请求
        2. 在批量推理前逐个添加请求
        3. 支持聊天模板格式化后的对话

        处理流程：
        1. 如果prompt是字符串，使用分词器编码为token ID
        2. 创建Sequence对象封装请求信息
        3. 将Sequence添加到调度器的等待队列
        """
        # 如果prompt是字符串，使用分词器编码
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)

        # 创建Sequence对象封装请求
        seq = Sequence(prompt, sampling_params)

        # 将请求添加到调度器
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
        # 1. 调度器选择序列批次
        # 返回：seqs - 选中的序列列表，is_prefill - 是否为prefill阶段
        seqs, is_prefill = self.scheduler.schedule()

        # 2. 调用ModelRunner执行推理
        # 传递序列和阶段信息，返回生成的token ID列表
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 3. 调度器后处理
        # 将生成的token添加到序列中，更新状态
        self.scheduler.postprocess(seqs, token_ids)

        # 4. 收集已完成的序列
        # 只返回状态为FINISHED的序列
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # 5. 计算token统计
        # prefill阶段：统计处理的token数
        # decode阶段：负数表示序列数（用于计算decode吞吐量）
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        return outputs, num_tokens

    def is_finished(self):
        """
        检查是否所有请求都已完成。

        使用方法：
        while not llm.is_finished():
            output, num_tokens = llm.step()

        使用场景：
        1. 在generate()中判断是否继续循环
        2. 手动控制推理流程时检查完成状态
        3. 监控推理进度

        返回值：
        - True: 所有请求已完成
        - False: 还有未完成的请求

        原理：
        检查调度器的waiting队列和running队列是否都为空
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成文本的主接口。

        使用方法：
        # 单个采样参数
        outputs = llm.generate(
            ["你好", "介绍一下北京"],
            SamplingParams(temperature=0.7, max_tokens=100)
        )

        # 多个采样参数（每个prompt对应一个）
        outputs = llm.generate(
            ["prompt1", "prompt2"],
            [SamplingParams(temperature=0.7), SamplingParams(temperature=0.9)]
        )

        参数：
        - prompts: 输入提示列表，可以是字符串列表或token ID列表
        - sampling_params: 采样参数，可以是单个参数（所有prompt共用）或参数列表
        - use_tqdm: 是否显示进度条，默认为True

        返回值：
        列表，每个元素包含：
        - "text": 解码后的文本
        - "token_ids": 生成的token ID列表

        使用场景：
        1. 批量文本生成
        2. 对话系统
        3. 代码生成
        4. 文本续写

        性能优化：
        - prefill阶段：批量处理所有prompt的输入token，高吞吐量
        - decode阶段：逐个生成新token，低延迟
        """
        # 初始化进度条（如果启用）
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 将单个采样参数扩展为列表（如果需要）
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加所有请求到调度器
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 用于存储所有输出结果
        outputs = {}

        # 吞吐量统计
        prefill_throughput = decode_throughput = 0.

        # 主推理循环
        while not self.is_finished():
            # 记录开始时间
            t = perf_counter()

            # 执行一次推理步骤
            output, num_tokens = self.step()

            # 更新进度条和吞吐量统计
            if use_tqdm:
                if num_tokens > 0:
                    # prefill阶段：计算处理token的吞吐量
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # decode阶段：计算生成token的吞吐量（负号转换为正数）
                    decode_throughput = -num_tokens / (perf_counter() - t)

                # 更新进度条后缀，显示实时吞吐量
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            # 收集完成的序列输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # 按seq_id排序，确保输出顺序与输入顺序一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 解码token为文本，并格式化输出
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 关闭进度条
        if use_tqdm:
            pbar.close()

        return outputs
