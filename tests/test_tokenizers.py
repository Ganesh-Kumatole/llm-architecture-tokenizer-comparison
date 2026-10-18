import json
from collections import Counter

import pytest

from src.tokenization import (BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID,
                              CharTokenizer, WordTokenizer, load_tokenizer, normalize)

CORPUS = [
    'One day, a little girl named Lily found a needle.\n\nShe said, "Mom, I found this!"',
    "Tom didn\u00e2\u20ac\u2122t like the rain. It was cold.  Very cold!",
    "The end.",
]


def build(kind, **kw):
    return CharTokenizer(**kw) if kind == "char" else WordTokenizer(max_vocab=1000, **kw)


@pytest.fixture(params=["char", "word"])
def kind(request):
    return request.param


@pytest.fixture
def tok(kind):
    return build(kind).fit(CORPUS)


# ---------------------------------------------------------------- shared contract
def test_special_tokens_have_fixed_ids(tok):
    assert (PAD_ID, UNK_ID, BOS_ID, EOS_ID) == (0, 1, 2, 3)
    assert tok.vocab[:4] == list(SPECIAL_TOKENS)


def test_roundtrip_exact_for_known_text(tok):
    for s in CORPUS:
        assert tok.decode(tok.encode(s)) == normalize(s)


def test_empty_string(tok):
    assert tok.encode("") == []
    assert tok.encode("", add_bos=True, add_eos=True) == [BOS_ID, EOS_ID]
    assert tok.decode([]) == ""
    assert tok.decode([BOS_ID, EOS_ID]) == ""


def test_bos_eos_flags(tok):
    ids = tok.encode("The end.", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID
    assert tok.encode("The end.") == ids[1:-1]
    assert tok.decode(ids) == "The end."
    full = tok.decode(ids, skip_special=False)
    assert full.startswith("<bos>") and full.endswith("<eos>")


def test_special_token_strings_in_text_are_not_parsed(tok):
    ids = tok.encode("<bos>hello<eos><pad>")
    assert not ({PAD_ID, BOS_ID, EOS_ID} & set(ids))


def test_unknown_input_maps_to_unk_and_is_visible(tok):
    ids = tok.encode("Zebra \u00b6 \u65e5\u672c")
    assert UNK_ID in ids
    assert "<unk>" in tok.decode(ids)


def test_whitespace_only_input_is_handled(tok):
    ids = tok.encode(" \n ")
    assert ids and all(0 <= i < tok.vocab_size for i in ids)


def test_decode_rejects_out_of_range_ids(tok):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])
    with pytest.raises(ValueError):
        tok.decode([-1])


def test_encode_before_fit_raises():
    with pytest.raises(RuntimeError):
        CharTokenizer().encode("x")
    with pytest.raises(RuntimeError):
        WordTokenizer().encode("x")


def test_normalization_happens_inside_encode(tok):
    assert tok.encode("didn\u00e2\u20ac\u2122t") == tok.encode("didn\u2019t")


def test_fit_is_deterministic_and_order_independent(kind, tok):
    assert build(kind).fit(reversed(CORPUS)).vocab == tok.vocab


def test_save_reload_equivalence(tok, tmp_path):
    tok.save(tmp_path)
    again = load_tokenizer(tmp_path)
    assert type(again) is type(tok)
    assert again.vocab == tok.vocab
    for s in CORPUS + ["", "Zebra \u00b6", " \n ", "didn\u00e2\u20ac\u2122t"]:
        assert again.encode(s, add_bos=True, add_eos=True) == tok.encode(s, add_bos=True, add_eos=True)
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["vocab_size"] == tok.vocab_size
    assert cfg["provenance"]["n_fit_stories"] == len(CORPUS)
    assert cfg["normalization"]["repair_mojibake"] is True


def test_saved_files_are_byte_identical_across_refits(kind, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build(kind).fit(CORPUS).save(a)
    build(kind).fit(reversed(CORPUS)).save(b)
    for name in ("vocab.json", "config.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_tampered_vocab_is_detected(tok, tmp_path):
    tok.save(tmp_path)
    vocab = json.loads((tmp_path / "vocab.json").read_text(encoding="utf-8"))
    vocab.append("TAMPERED")
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokenizer(tmp_path)


def test_repair_flag_changes_vocab_and_is_recorded(kind, tmp_path):
    t = build(kind, repair_mojibake=False).fit(CORPUS)
    assert any("\u00e2" in tk for tk in t.vocab)
    t.save(tmp_path)
    assert load_tokenizer(tmp_path).repair_mojibake is False


# ---------------------------------------------------------------- character-specific
def test_char_vocab_is_specials_plus_sorted_unique_chars():
    t = CharTokenizer().fit(CORPUS)
    expected = sorted(set().union(*(set(normalize(s)) for s in CORPUS)))
    assert t.vocab == list(SPECIAL_TOKENS) + expected
    assert "\u2019" in t.vocab and "\u00e2" not in t.vocab


# ---------------------------------------------------------------- word-specific
@pytest.mark.parametrize("s", [
    "Hello, world!", "  leading", "trailing  ", "a\n\nb", "a \n b",
    "tabs\tand\u00a0nbsp", "x_1 y2", "\u201cquoted\u201d", "(paren)", "end.\n", "", "   ",
])
def test_pretokenizer_partitions_text_exactly(s):
    assert "".join(WordTokenizer().tokenize(s)) == normalize(s)


def test_pretokenizer_attaches_single_leading_space():
    t = WordTokenizer()
    assert t.tokenize("Hello, world!") == ["Hello", ",", " world", "!"]
    assert t.tokenize("end.\n\nLily") == ["end", ".", "\n\n", "Lily"]
    assert t.tokenize("didn\u2019t") == ["didn", "\u2019", "t"]


def test_word_vocab_cap_is_exact_and_frequency_ordered():
    t = WordTokenizer(max_vocab=10).fit(CORPUS)
    assert t.vocab_size == 10
    counts = Counter(w for s in CORPUS for w in t.tokenize(s))
    kept = t.vocab[4:]
    assert [counts[w] for w in kept] == sorted((counts[w] for w in kept), reverse=True)
    dropped = [c for w, c in counts.items() if w not in set(kept)]
    assert max(dropped) <= min(counts[w] for w in kept)


def test_word_min_count_drops_rare_tokens():
    t = WordTokenizer(max_vocab=1000, min_count=2).fit(CORPUS)
    counts = Counter(w for s in CORPUS for w in t.tokenize(s))
    assert all(counts[w] >= 2 for w in t.vocab[4:])
    assert " Lily" not in t.vocab and any(c < 2 for c in counts.values())


def test_word_fit_coverage_is_recorded():
    assert WordTokenizer(max_vocab=1000).fit(CORPUS).provenance["fit_token_coverage"] == 1.0
    assert 0.0 < WordTokenizer(max_vocab=10).fit(CORPUS).provenance["fit_token_coverage"] < 1.0


def test_word_rejects_tiny_vocab_cap():
    with pytest.raises(ValueError):
        WordTokenizer(max_vocab=4)


def test_word_unseen_word_is_a_single_unk_that_swallows_its_space():
    t = WordTokenizer(max_vocab=1000).fit(CORPUS)
    ids = t.encode("The end. Zebra")
    assert ids.count(UNK_ID) == 1
    assert t.decode(ids) == "The end.<unk>"
