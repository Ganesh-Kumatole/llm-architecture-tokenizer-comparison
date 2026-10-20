import json

import pytest

from src.tokenization import (
    BPETokenizer,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    load_tokenizer,
    normalize,
)

CORPUS = [
    'One day, a little girl named Lily found a needle.\n\n'
    'She said, "Mom, I found this!"',
    "Tom didn\u00e2\u20ac\u2122t like the rain. It was cold. Very cold!",
    "The end.",
]


@pytest.fixture
def bpe_tokenizer():
    return BPETokenizer(
        target_vocab_size=512,
        min_frequency=1,
    ).fit(CORPUS, provenance={"source": "unit-test"})


def test_bpe_special_ids_are_fixed(bpe_tokenizer):
    assert (PAD_ID, UNK_ID, BOS_ID, EOS_ID) == (0, 1, 2, 3)
    assert bpe_tokenizer.vocab[:4] == list(SPECIAL_TOKENS)


def test_bpe_vocab_ids_are_in_range(bpe_tokenizer):
    assert bpe_tokenizer.vocab_size == len(bpe_tokenizer.vocab)
    assert bpe_tokenizer.vocab_size >= 260


def test_bpe_roundtrip_exact_for_normalized_text(bpe_tokenizer):
    samples = CORPUS + [
        "Hello, world!  \n",
        "cafe\u0301",
        "日本語 and emoji 🦊",
        "leading spaces  and\ttabs",
    ]
    for text in samples:
        assert bpe_tokenizer.decode(
            bpe_tokenizer.encode(text)
        ) == normalize(text)


def test_bpe_empty_string(bpe_tokenizer):
    assert bpe_tokenizer.encode("") == []
    assert bpe_tokenizer.decode([]) == ""
    assert bpe_tokenizer.encode(
        "", add_bos=True, add_eos=True
    ) == [BOS_ID, EOS_ID]


def test_bpe_bos_eos_flags(bpe_tokenizer):
    text = "The end."
    ids = bpe_tokenizer.encode(text, add_bos=True, add_eos=True)

    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert bpe_tokenizer.decode(ids) == text
    assert bpe_tokenizer.decode(
        ids, skip_special=False
    ) == f"<bos>{text}<eos>"


def test_literal_special_strings_are_ordinary_text(bpe_tokenizer):
    text = "<bos>hello<eos><pad><unk>"
    ids = bpe_tokenizer.encode(text)

    # Raw text must not be parsed as reserved IDs.
    assert all(i >= 4 for i in ids)
    assert bpe_tokenizer.decode(ids) == text


def test_byte_level_bpe_handles_unseen_unicode(bpe_tokenizer):
    text = "Zebra ¶ 日本語 🦊"
    ids = bpe_tokenizer.encode(text)

    assert all(i >= 4 for i in ids)
    assert UNK_ID not in ids
    assert bpe_tokenizer.decode(ids) == text


def test_encode_before_fit_raises():
    with pytest.raises(RuntimeError):
        BPETokenizer().encode("text")


def test_empty_training_iterable_is_rejected():
    with pytest.raises(ValueError):
        BPETokenizer(target_vocab_size=512).fit([])


def test_bpe_fit_is_order_independent():
    forward = BPETokenizer(
        target_vocab_size=512, min_frequency=1
    ).fit(CORPUS)
    reverse = BPETokenizer(
        target_vocab_size=512, min_frequency=1
    ).fit(reversed(CORPUS))

    assert forward.vocab == reverse.vocab


def test_bpe_save_reload_equivalence(bpe_tokenizer, tmp_path):
    bpe_tokenizer.save(tmp_path)

    for filename in ("tokenizer.json", "vocab.json", "config.json"):
        assert (tmp_path / filename).is_file()

    loaded = load_tokenizer(tmp_path)
    assert isinstance(loaded, BPETokenizer)
    assert loaded.vocab == bpe_tokenizer.vocab

    samples = CORPUS + ["", "日本語 🦊", "<eos> is ordinary text"]
    for text in samples:
        assert loaded.encode(
            text, add_bos=True, add_eos=True
        ) == bpe_tokenizer.encode(
            text, add_bos=True, add_eos=True
        )
        assert loaded.decode(loaded.encode(text)) == normalize(text)

    cfg = json.loads(
        (tmp_path / "config.json").read_text(encoding="utf-8")
    )
    assert cfg["provenance"]["n_fit_stories"] == len(CORPUS)
    assert cfg["normalization"]["repair_mojibake"] is True


def test_bpe_tampered_vocab_is_detected(bpe_tokenizer, tmp_path):
    bpe_tokenizer.save(tmp_path)
    path = tmp_path / "vocab.json"

    vocab = json.loads(path.read_text(encoding="utf-8"))
    vocab.append("TAMPERED")
    path.write_text(
        json.dumps(vocab, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vocab.json hash mismatch"):
        load_tokenizer(tmp_path)


def test_bpe_tampered_backend_is_detected(bpe_tokenizer, tmp_path):
    bpe_tokenizer.save(tmp_path)
    path = tmp_path / "tokenizer.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tokenizer.json hash mismatch"):
        load_tokenizer(tmp_path)
