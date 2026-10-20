"""Build tokenizers from YAML; load artifacts by recorded tokenizer kind."""
from pathlib import Path

import yaml

from .base import read_json
from .bpe import BPETokenizer
from .char import CharTokenizer
from .normalization import NORMALIZATION_VERSION
from .word import WordTokenizer

REGISTRY = {
    CharTokenizer.kind: CharTokenizer,
    WordTokenizer.kind: WordTokenizer,
    BPETokenizer.kind: BPETokenizer,
}


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(tokenizer_cfg: dict, data_cfg: dict):
    kind = tokenizer_cfg["type"]
    if kind not in REGISTRY:
        raise ValueError(
            f"unknown tokenizer type {kind!r}; "
            f"known: {sorted(REGISTRY)}"
        )

    norm = data_cfg["normalization"]
    if norm["version"] != NORMALIZATION_VERSION:
        raise ValueError(
            f"config normalization {norm['version']!r} "
            f"!= code {NORMALIZATION_VERSION!r}"
        )

    params = {
        k: v for k, v in tokenizer_cfg.items()
        if k not in ("type", "artifact_dir")
    }
    return REGISTRY[kind](
        repair_mojibake=norm["repair_mojibake"],
        **params,
    )


def load_tokenizer(directory):
    directory = Path(directory)
    kind = read_json(directory / "config.json")["kind"]

    if kind not in REGISTRY:
        raise ValueError(
            f"unknown tokenizer kind {kind!r} in artifact"
        )
    return REGISTRY[kind].load(directory)
