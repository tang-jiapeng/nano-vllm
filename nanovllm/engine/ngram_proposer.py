"""N-gram proposer for speculative decoding.

The proposer searches the current token history for the longest suffix n-gram
that appeared previously and reuses the following tokens as draft candidates.
This mirrors the prompt-lookup / n-gram style speculative decoding path used in
vLLM V1, but keeps the implementation intentionally lightweight.
"""


class NgramProposer:
    """Draft proposer based on repeated n-grams in the current token history."""

    def __init__(self, min_ngram: int = 1, max_ngram: int = 4):
        assert min_ngram >= 1
        assert max_ngram >= min_ngram
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram

    def propose(self, token_ids: list[int], max_proposal_len: int) -> list[int]:
        """Return up to ``max_proposal_len`` draft tokens from prior n-gram matches."""
        if max_proposal_len <= 0:
            return []

        token_count = len(token_ids)
        max_ngram = min(self.max_ngram, token_count - 1)
        if max_ngram < self.min_ngram:
            return []

        for ngram_size in range(max_ngram, self.min_ngram - 1, -1):
            suffix = token_ids[token_count - ngram_size :]
            search_limit = token_count - ngram_size

            for start in range(search_limit):
                if token_ids[start : start + ngram_size] != suffix:
                    continue

                draft_start = start + ngram_size
                draft_end = min(draft_start + max_proposal_len, token_count)
                if draft_start < draft_end:
                    return token_ids[draft_start:draft_end]

        return []
