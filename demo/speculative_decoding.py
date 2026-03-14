"""
nano-vllm N-gram 推测解码 (Speculative Decoding) 示例
====================================================

当前实现采用更轻量的 N-gram proposer 路线：

- 不再依赖额外 draft model
- 直接从 prompt / 已生成上下文中查找可复用的 n-gram
- 用 target model 一次 verify 多个 draft token
"""

import argparse
import os
from time import perf_counter

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def build_prompts(tokenizer):
    prompts = [
        "Repeat the following Python pattern and continue it with a short explanation:\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "Continue this markdown table and summarize the pattern briefly:\n"
        "| step | action |\n"
        "|------|--------|\n"
        "| 1    | load   |\n"
        "| 2    | parse  |\n"
        "| 3    | load   |\n"
        "| 4    | parse  |\n",
        "请延续下面这种重复结构，并给出一个简短总结：\n"
        "第一步：读取配置。\n"
        "第二步：校验输入。\n"
        "第三步：读取配置。\n"
        "第四步：校验输入。\n",
    ]
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]


def make_llm(model_path, args, speculative_method):
    return LLM(
        model_path,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(args.max_model_len, 4096),
        speculative_method=speculative_method,
        num_speculative_tokens=(
            args.num_speculative_tokens if speculative_method is not None else 0
        ),
        ngram_prompt_lookup_min=args.ngram_prompt_lookup_min,
        ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
    )


def run_case(model_path, prompts, sampling_params, args, speculative_method):
    llm = make_llm(model_path, args, speculative_method)
    start = perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = perf_counter() - start
    llm.exit()
    return outputs, elapsed


def print_outputs(title, prompts, outputs, elapsed):
    print(f"\n{'=' * 72}")
    print(title)
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"{'=' * 72}")
    total_proposed = 0
    total_accepted = 0
    for prompt, output in zip(prompts, outputs):
        print("\n" + "-" * 60)
        print(f"Prompt:     {prompt[:80]!r}...")
        print(f"Completion: {output['text']!r}")
        proposed = output.get("proposed", 0)
        accepted = output.get("accepted", 0)
        total_proposed += proposed
        total_accepted += accepted
        if proposed > 0:
            print(f"Accept rate: {accepted}/{proposed} = {accepted/proposed:.2%}")

    if total_proposed > 0:
        print("\n" + "-" * 60)
        print(
            f"Overall accept rate: {total_accepted}/{total_proposed} "
            f"= {total_accepted/total_proposed:.2%}"
        )
    else:
        print("\n(Speculative decoding not enabled — no accept rate to report)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="nano-vllm n-gram speculative decoding example"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/Qwen3-1.7B",
        help="目标模型路径",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        "-K",
        type=int,
        default=5,
        help="每轮推测的候选 token 数",
    )
    parser.add_argument(
        "--ngram-prompt-lookup-min",
        type=int,
        default=2,
        help="N-gram proposer 的最小匹配长度",
    )
    parser.add_argument(
        "--ngram-prompt-lookup-max",
        type=int,
        default=4,
        help="N-gram proposer 的最大匹配长度",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="先跑 baseline，再跑 n-gram speculative decoding 做对比",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="禁用 CUDA Graph，方便调试",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = build_prompts(tokenizer)
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    if args.compare_baseline:
        baseline_outputs, baseline_elapsed = run_case(
            model_path, prompts, sampling_params, args, speculative_method=None
        )
        print_outputs(
            "Baseline decode (no speculative decoding)",
            prompts,
            baseline_outputs,
            baseline_elapsed,
        )

    speculative_outputs, speculative_elapsed = run_case(
        model_path, prompts, sampling_params, args, speculative_method="ngram"
    )
    print_outputs(
        "N-gram speculative decoding",
        prompts,
        speculative_outputs,
        speculative_elapsed,
    )
