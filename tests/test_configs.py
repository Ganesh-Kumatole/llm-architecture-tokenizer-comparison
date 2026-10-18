from pathlib import Path

import pytest

from src.tokenization import SPECIAL_TOKENS, build_tokenizer, load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_data_config_matches_code_contract():
    d = load_yaml(ROOT / "configs" / "data.yaml")
    assert d["normalization"]["version"] == "norm-v1"
    assert d["special_tokens"] == list(SPECIAL_TOKENS)
    assert d["data"]["max_context_tokens"] == 128


@pytest.mark.parametrize("name", ["character", "word"])
def test_tokenizer_configs_build(name):
    d = load_yaml(ROOT / "configs" / "data.yaml")
    t = load_yaml(ROOT / "configs" / f"tokenizer_{name}.yaml")
    tok = build_tokenizer(t, d)
    assert tok.kind == t["type"] == name
    assert t["artifact_dir"] == f"artifacts/tokenizers/{name}"
