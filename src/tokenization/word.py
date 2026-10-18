r"""Word-level tokenizer with whitespace-attached pre-tokens (GPT-2 style).

Pre-tokenizer regex:   ` ?\w+`  |  ` ?[^\w\s]`  |  `\s+`
  * a word with ONE optional leading space      "Lily", " world"
  * one punctuation/symbol char, optional space ",", " ("
  * any other whitespace run                    "\n\n", "  "
Every character of the normalized text belongs to exactly one pre-token, so
"".join(pre_tokens) == normalized text. decode(encode(x)) == normalize(x) whenever no
token is out-of-vocabulary. An out-of-vocabulary pre-token becomes ONE <unk>, which also
swallows its leading space (documented limitation; the UNK rate is reported).
Vocabulary = top (max_vocab - 4) pre-tokens by training frequency, ties broken by the
token string, so fitting is deterministic and independent of data order.
Case is preserved ("The" and "the" are different entries) to match the shared
normalization policy.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .base import VocabTokenizer
from .normalization import normalize
from .specials import NUM_SPECIAL, SPECIAL_TOKENS

_PRETOKEN_RE = re.compile(r" ?\w+| ?[^\w\s]|\s+")


class WordTokenizer(VocabTokenizer):
    kind = "word"

    def __init__(self, max_vocab: int = 10_000, min_count: int = 1, repair_mojibake: bool = True):
        if max_vocab <= NUM_SPECIAL:
            raise ValueError(f"max_vocab must exceed the {NUM_SPECIAL} special tokens")
        if min_count < 1:
            raise ValueError("min_count must be >= 1")
        super().__init__(repair_mojibake)
        self.max_vocab = max_vocab
        self.min_count = min_count

    def _split(self, text: str) -> list[str]:
        return _PRETOKEN_RE.findall(text)

    def _extra_params(self) -> dict:
        return {
            "max_vocab": self.max_vocab,
            "min_count": self.min_count,
            "pretokenizer_regex": _PRETOKEN_RE.pattern,
        }

    def fit(self, texts: Iterable[str], provenance: dict | None = None) -> "WordTokenizer":
        counts: Counter = Counter()
        n = 0
        for t in texts:
            counts.update(self._split(normalize(t, self.repair_mojibake)))
            n += 1
        total = sum(counts.values())
        eligible = sorted(
            ((tok, c) for tok, c in counts.items() if c >= self.min_count),
            key=lambda kv: (-kv[1], kv[0]),
        )
        kept = eligible[: self.max_vocab - NUM_SPECIAL]
        self._set_vocab(list(SPECIAL_TOKENS) + [tok for tok, _ in kept])
        covered = sum(c for _, c in kept)
        self.provenance = {
            **(provenance or {}),
            "n_fit_stories": n,
            "n_distinct_pretokens_in_fit": len(counts),
            "fit_token_coverage": covered / total if total else 0.0,
        }
        return self

    @classmethod
    def _from_config(cls, cfg: dict) -> "WordTokenizer":
        p = cfg["params"]
        if p["pretokenizer_regex"] != _PRETOKEN_RE.pattern:
            raise ValueError("artifact was built with a different pre-tokenizer regex")
        return cls(
            max_vocab=p["max_vocab"],
            min_count=p["min_count"],
            repair_mojibake=cfg["normalization"]["repair_mojibake"],
        )
