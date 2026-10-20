"""Byte-level BPE tokenizer satisfying the project's BaseTokenizer contract.

External IDs:
    PAD=0, UNK=1, BOS=2, EOS=3
    Hugging Face BPE IDs are offset by 4.

Special tokens are NOT registered as added tokens in the Hugging Face backend.
Consequently, literal strings such as '<eos>' in source text remain ordinary text.
"""
from __future__ import annotations

import hashlib
import platform
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .base import BaseTokenizer, read_json, vocab_sha256, write_json
from .normalization import (
    NORMALIZATION_VERSION,
    normalization_config,
    normalize,
)
from .specials import (
    BOS_ID,
    EOS_ID,
    NUM_SPECIAL,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
)


class BPETokenizer(BaseTokenizer):
    """Byte-level BPE with the common project tokenizer interface."""

    kind = "bpe"
    ID_OFFSET = NUM_SPECIAL

    def __init__(
        self,
        target_vocab_size: int = 8000,
        min_frequency: int = 2,
        repair_mojibake: bool = True,
    ):
        if target_vocab_size < NUM_SPECIAL + 256:
            raise ValueError(
                "target_vocab_size must be at least 260 for "
                "four reserved IDs plus the 256-byte alphabet"
            )
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")

        super().__init__(repair_mojibake=repair_mojibake)
        self.target_vocab_size = target_vocab_size
        self.min_frequency = min_frequency
        self._backend: Tokenizer | None = None

    def _require_fitted(self) -> None:
        if self._backend is None:
            raise RuntimeError(
                "BPETokenizer is not fitted; call fit() or load()"
            )

    @property
    def vocab_size(self) -> int:
        self._require_fitted()
        return self._backend.get_vocab_size() + self.ID_OFFSET

    def _vocab_tokens(self) -> list[str]:
        """Vocabulary strings ordered by their externally visible IDs."""
        self._require_fitted()
        raw_vocab = self._backend.get_vocab()
        pieces = [
            token
            for token, _ in sorted(
                raw_vocab.items(), key=lambda item: item[1]
            )
        ]
        return list(SPECIAL_TOKENS) + pieces

    @property
    def vocab(self) -> list[str]:
        return self._vocab_tokens()

    def fit(
        self,
        texts: Iterable[str],
        provenance: dict | None = None,
    ) -> "BPETokenizer":
        """Train on the iterable supplied by the caller (training split only)."""

        backend = Tokenizer(models.BPE(unk_token=None))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )
        backend.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size - self.ID_OFFSET,
            min_frequency=self.min_frequency,
            special_tokens=[],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )

        n_stories = 0

        def normalized_texts():
            nonlocal n_stories
            for text in texts:
                n_stories += 1
                yield normalize(text, self.repair_mojibake)

        backend.train_from_iterator(normalized_texts(), trainer=trainer)

        if n_stories == 0:
            raise ValueError("Cannot fit BPE on an empty text iterable")

        self._backend = backend
        self.provenance = {
            **(provenance or {}),
            "n_fit_stories": n_stories,
            "actual_vocab_size": self.vocab_size,
        }
        return self

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        self._require_fitted()

        normalized = normalize(text, self.repair_mojibake)
        backend_ids = self._backend.encode(normalized).ids
        ids = [i + self.ID_OFFSET for i in backend_ids]

        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)

        return ids

    def decode(
        self,
        ids: Sequence[int],
        skip_special: bool = True,
    ) -> str:
        """Decode BPE IDs while following the shared special-token policy."""

        self._require_fitted()
        vocab_size = self.vocab_size
        output: list[str] = []
        pending_bpe_ids: list[int] = []

        def flush_bpe():
            if pending_bpe_ids:
                raw_ids = [
                    int(i) - self.ID_OFFSET for i in pending_bpe_ids
                ]
                output.append(self._backend.decode(raw_ids))
                pending_bpe_ids.clear()

        for raw_id in ids:
            token_id = int(raw_id)

            if not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"token id {token_id} outside vocabulary "
                    f"[0, {vocab_size})"
                )

            if token_id >= self.ID_OFFSET:
                pending_bpe_ids.append(token_id)
                continue

            flush_bpe()

            if token_id == UNK_ID:
                output.append("<unk>")
            elif not skip_special:
                output.append(SPECIAL_TOKENS[token_id])

        flush_bpe()
        return "".join(output)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save(self, directory) -> None:
        self._require_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        tokenizer_path = directory / "tokenizer.json"
        vocab_path = directory / "vocab.json"
        config_path = directory / "config.json"

        self._backend.save(str(tokenizer_path))
        vocab = self._vocab_tokens()
        write_json(vocab_path, vocab)

        cfg = {
            "kind": self.kind,
            "vocab_size": len(vocab),
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "special_ids": {
                "pad": PAD_ID,
                "unk": UNK_ID,
                "bos": BOS_ID,
                "eos": EOS_ID,
            },
            "normalization": normalization_config(
                self.repair_mojibake
            ),
            "params": {
                "target_vocab_size": self.target_vocab_size,
                "min_frequency": self.min_frequency,
                "id_offset": self.ID_OFFSET,
                "algorithm": "huggingface-tokenizers-byte-level-bpe",
            },
            "provenance": self.provenance,
            "environment": {
                "python": platform.python_version(),
                "tokenizers": package_version("tokenizers"),
            },
            "vocab_sha256": vocab_sha256(vocab),
            "tokenizer_json_sha256": self._sha256(tokenizer_path),
        }
        write_json(config_path, cfg)

    @classmethod
    def load(cls, directory) -> "BPETokenizer":
        directory = Path(directory)
        cfg = read_json(directory / "config.json")
        vocab = read_json(directory / "vocab.json")
        tokenizer_path = directory / "tokenizer.json"

        if cfg["kind"] != cls.kind:
            raise ValueError("Artifact is not a BPE tokenizer")
        if cfg["special_tokens"] != list(SPECIAL_TOKENS):
            raise ValueError("Special-token contract mismatch")
        if cfg["normalization"]["version"] != NORMALIZATION_VERSION:
            raise ValueError(
                "Normalization version mismatch; refit the tokenizer"
            )
        if cfg["params"]["id_offset"] != cls.ID_OFFSET:
            raise ValueError("BPE ID-offset contract mismatch")
        if vocab_sha256(vocab) != cfg["vocab_sha256"]:
            raise ValueError("vocab.json hash mismatch")
        if cls._sha256(tokenizer_path) != cfg["tokenizer_json_sha256"]:
            raise ValueError("tokenizer.json hash mismatch")

        obj = cls(
            target_vocab_size=cfg["params"]["target_vocab_size"],
            min_frequency=cfg["params"]["min_frequency"],
            repair_mojibake=cfg["normalization"][
                "repair_mojibake"
            ],
        )
        obj._backend = Tokenizer.from_file(str(tokenizer_path))
        if obj.vocab_size != cfg["vocab_size"]:
            raise ValueError("Stored vocabulary size does not match model")
        obj.provenance = cfg.get("provenance", {})
        return obj
