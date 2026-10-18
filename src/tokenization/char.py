"""Character-level tokenizer: one token per Unicode code point of the normalized text."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .base import VocabTokenizer
from .normalization import normalize
from .specials import SPECIAL_TOKENS


class CharTokenizer(VocabTokenizer):
    kind = "character"

    def _split(self, text: str) -> list[str]:
        return list(text)

    def fit(self, texts: Iterable[str], provenance: dict | None = None) -> "CharTokenizer":
        counts: Counter = Counter()
        n = 0
        for t in texts:
            counts.update(normalize(t, self.repair_mojibake))
            n += 1
        # Deterministic: specials first, then every training character in code-point order.
        self._set_vocab(list(SPECIAL_TOKENS) + sorted(counts))
        self.provenance = {**(provenance or {}), "n_fit_stories": n}
        return self

    @classmethod
    def _from_config(cls, cfg: dict) -> "CharTokenizer":
        return cls(repair_mojibake=cfg["normalization"]["repair_mojibake"])
