import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams


def run_bench(
    model: str,
    prompts: list[str],
    sampling_params: list[SamplingParams],
    enforce_eager: bool,
    chunked_prefill: bool,
    speculative_method: str | None = None,
    num_speculative_tokens: int = 0,
    ngram_prompt_lookup_min: int = 2,
    ngram_prompt_lookup_max: int = 4,
):
    llm = LLM(
        model,
        enforce_eager=enforce_eager,
        max_model_len=4096,
        chunked_prefill=chunked_prefill,
        speculative_method=speculative_method,
        num_speculative_tokens=num_speculative_tokens,
        ngram_prompt_lookup_min=ngram_prompt_lookup_min,
        ngram_prompt_lookup_max=ngram_prompt_lookup_max,
    )

    llm.generate(["Benchmark: "], SamplingParams(temperature=0.6, max_tokens=8))

    t = time.time()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t

    proposed = sum(out.get("proposed", 0) for out in outputs)
    accepted = sum(out.get("accepted", 0) for out in outputs)

    llm.exit()
    return total_tokens, t, throughput, proposed, accepted


def main():
    seed(0)
    num_seqs = 256
    max_output_len = 1024
    target_path = os.path.expanduser("~/nano-vllm/models/Qwen3-1.7B/")
    num_speculative_tokens = 5

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=target_path)
    parser.add_argument(
        "--num-speculative-tokens", type=int, default=num_speculative_tokens
    )
    parser.add_argument("--num-seqs", type=int, default=num_seqs)
    parser.add_argument("--max-output", type=int, default=max_output_len)
    parser.add_argument("--ngram-prompt-lookup-min", type=int, default=2)
    parser.add_argument("--ngram-prompt-lookup-max", type=int, default=4)
    parser.add_argument("--chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    base_lines = [
        "def fibonacci(n):",
        "    if n <= 1:",
        "        return n",
        "    return fibonacci(n - 1) + fibonacci(n - 2)",
        "",
        "def main():",
        "    values = [fibonacci(i) for i in range(10)]",
        "    print(values)",
        "",
        "if __name__ == '__main__':",
        "    main()",
    ]
    repeated_prompt = "\n".join(base_lines * 24)
    prompts = [
        repeated_prompt + f"\n# Variant {i}\n# Continue the code and explain it step by step.\n"
        for i in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.0,
            ignore_eos=True,
            max_tokens=randint(100, args.max_output),
        )
        for _ in range(args.num_seqs)
    ]

    print("[1/2] Benchmark baseline (no speculative decoding)...")
    b_total, b_time, b_tps, _, _ = run_bench(
        model=args.model,
        prompts=prompts,
        sampling_params=sampling_params,
        enforce_eager=args.enforce_eager,
        chunked_prefill=args.chunked_prefill,
        speculative_method=None,
        num_speculative_tokens=0,
    )

    print("[2/2] Benchmark n-gram speculative decoding...")
    s_total, s_time, s_tps, proposed, accepted = run_bench(
        model=args.model,
        prompts=prompts,
        sampling_params=sampling_params,
        enforce_eager=args.enforce_eager,
        chunked_prefill=args.chunked_prefill,
        speculative_method="ngram",
        num_speculative_tokens=args.num_speculative_tokens,
        ngram_prompt_lookup_min=args.ngram_prompt_lookup_min,
        ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
    )

    mode = "chunked" if args.chunked_prefill else "standard"
    speedup = s_tps / b_tps
    accept_rate = accepted / proposed if proposed > 0 else 0.0

    print("\n=== Result ===")
    print(
        f"Baseline    | Total: {b_total}tok, Time: {b_time:.2f}s, Throughput: {b_tps:.2f}tok/s ({mode})"
    )
    print(
        f"Speculative | Total: {s_total}tok, Time: {s_time:.2f}s, Throughput: {s_tps:.2f}tok/s ({mode})"
    )
    print(
        f"Accept rate: {accepted}/{proposed} = {accept_rate:.2%}, Speedup: {speedup:.3f}x"
    )


if __name__ == "__main__":
    main()
