"""Top-level package exports for nano-vllm.

Use lazy imports here so lightweight CPU-side unit tests can import engine
submodules without immediately triggering model / Triton initialization.
"""

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name: str):
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
