"""Tokenizer contract + shared machinery for vocabulary-list tokenizers.

Contract (all tokenizers):
  fit(texts)            learn from TRAINING stories only (caller's responsibility; the
                        fitting scripts read train.parquet only and record its hash)
  encode(text, add_bos, add_eos) -> list[int]
                        normalizes `text` first; adds no special tokens unless asked
  decode(ids, skip_special=True) -> str
                        skips <pad>/<bos>/<eos>; <unk> is ALWAYS rendered literally
  save(dir) / load(dir) deterministic artifacts; load verifies a vocabulary hash
Special tokens are never parsed out of raw text: the literal string '<eos>' in a story
is ordinary text.
"""
from __future__ import annotations

import hashlib
import json
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Sequence

from .normalization import NORMALIZATION_VERSION, normalization_config, normalize
from .specials import BOS_ID, EOS_ID, NUM_SPECIAL, PAD_ID, SPECIAL_TOKENS, UNK_ID


def write_json(path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vocab_sha256(tokens: list[str]) -> str:
    blob = json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class BaseTokenizer(ABC):
    kind: str = "base"

    def __init__(self, repair_mojibake: bool = True):
        self.repair_mojibake = repair_mojibake
        self.provenance: dict = {}

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def fit(self, texts: Iterable[str], provenance: dict | None = None) -> "BaseTokenizer": ...

    @abstractmethod
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str: ...

    @abstractmethod
    def save(self, directory) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory) -> "BaseTokenizer": ...


class VocabTokenizer(BaseTokenizer):
    """Tokenizers whose vocabulary is an ordered list of strings (index = token ID)."""

    def __init__(self, repair_mojibake: bool = True):
        super().__init__(repair_mojibake)
        self._id_to_token: list[str] | None = None
        self._token_to_id: dict[str, int] | None = None

    # -- subclass hooks
    @abstractmethod
    def _split(self, normalized_text: str) -> list[str]: ...

    @classmethod
    @abstractmethod
    def _from_config(cls, cfg: dict) -> "VocabTokenizer": ...

    def _extra_params(self) -> dict:
        return {}

    # -- vocabulary handling
    def _set_vocab(self, tokens: list[str]) -> None:
        if list(tokens[:NUM_SPECIAL]) != list(SPECIAL_TOKENS):
            raise ValueError("vocabulary must start with the special tokens in contract order")
        if len(set(tokens)) != len(tokens):
            raise ValueError("duplicate tokens in vocabulary")
        self._id_to_token = list(tokens)
        self._token_to_id = {t: i for i, t in enumerate(self._id_to_token)}

    def _require_fitted(self) -> None:
        if self._id_to_token is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted (call fit() or load())")

    @property
    def vocab_size(self) -> int:
        self._require_fitted()
        return len(self._id_to_token)

    @property
    def vocab(self) -> list[str]:
        self._require_fitted()
        return list(self._id_to_token)

    # -- encode / decode
    def tokenize(self, text: str) -> list[str]:
        """Normalize then split into token strings (no vocabulary needed)."""
        return self._split(normalize(text, self.repair_mojibake))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        self._require_fitted()
        ids = [self._token_to_id.get(t, UNK_ID) for t in self.tokenize(text)]
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        self._require_fitted()
        n = len(self._id_to_token)
        out = []
        for i in ids:
            if not 0 <= i < n:
                raise ValueError(f"token id {i} outside vocabulary [0, {n})")
            if i in (PAD_ID, BOS_ID, EOS_ID):
                if not skip_special:
                    out.append(self._id_to_token[i])
            else:
                out.append(self._id_to_token[i])   # UNK_ID renders as the literal '<unk>'
        return "".join(out)

    # -- persistence
    def config(self) -> dict:
        self._require_fitted()
        return {
            "kind": self.kind,
            "vocab_size": self.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "special_ids": {"pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID},
            "normalization": normalization_config(self.repair_mojibake),
            "params": self._extra_params(),
            "provenance": self.provenance,
            "environment": {"python": platform.python_version()},
            "vocab_sha256": vocab_sha256(self._id_to_token),
        }

    def save(self, directory) -> None:
        self._require_fitted()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "vocab.json", self._id_to_token)
        write_json(d / "config.json", self.config())

    @classmethod
    def load(cls, directory):
        d = Path(directory)
        cfg = read_json(d / "config.json")
        vocab = read_json(d / "vocab.json")
        if cfg["kind"] != cls.kind:
            raise ValueError(f"artifact kind {cfg['kind']!r} != {cls.kind!r}")
        if cfg["special_tokens"] != list(SPECIAL_TOKENS):
            raise ValueError("special-token contract mismatch")
        if cfg["normalization"]["version"] != NORMALIZATION_VERSION:
            raise ValueError("normalization version mismatch; refit the tokenizer")
        if vocab_sha256(vocab) != cfg["vocab_sha256"]:
            raise ValueError("vocab.json does not match the hash in config.json")
        tok = cls._from_config(cfg)
        tok._set_vocab(vocab)
        tok.provenance = cfg.get("provenance", {})
        return tok
