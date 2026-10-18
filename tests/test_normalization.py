import pytest

from src.tokenization import normalize


@pytest.mark.parametrize("raw, expected", [
    ("didn\u00e2\u20ac\u2122t", "didn\u2019t"),
    ("\u00e2\u20ac\u0153Hi\u00e2\u20ac she said", "\u201cHi\u201d she said"),   # bare close + space
    ("said \u00e2\u20ac\u0153Hi!\u00e2\u20ac", "said \u201cHi!\u201d"),          # bare close at end
    ("\u00e2\u20ac\u0153Hi!\u00e2\u20ac\nNext", "\u201cHi!\u201d\nNext"),        # bare close + newline
    ("\u00e2\u20ac\u00e2\u20ac\u0153", "\u201d\u201c"),                          # close directly before open
    ("\u00e2\u20ac\u02dcquoted\u00e2\u20ac\u2122", "\u2018quoted\u2019"),
    ("wait\u00e2\u20ac\u00a6", "wait\u2026"),
    ("1\u00e2\u20ac\u201c2", "1\u20132"),
    ("so\u00e2\u20ac\u201dthen", "so\u2014then"),
    ("Caf\u00c3\u00a9", "Caf\u00e9"),
    ("Espa\u00c3\u00b1a", "Espa\u00f1a"),
])
def test_mojibake_repairs(raw, expected):
    assert normalize(raw) == expected


def test_unrepairable_sequences_are_left_alone():
    s = "x\u00e2\u20ac\u2030y"
    assert normalize(s) == s


def test_clean_text_unchanged():
    s = 'She said, "Hi!" It\'s fine.\n\nThe end.'
    assert normalize(s) == s


def test_newlines_normalized_but_nothing_stripped_or_lowercased():
    assert normalize("  Hello\r\nWorld\r  ") == "  Hello\nWorld\n  "


def test_nfc_applied():
    assert normalize("e\u0301") == "\u00e9"


def test_repair_can_be_disabled():
    s = "didn\u00e2\u20ac\u2122t"
    assert normalize(s, repair=False) == s


def test_idempotent():
    s = "\u00e2\u20ac\u0153Caf\u00c3\u00a9\u00e2\u20ac\nok\r\n"
    assert normalize(normalize(s)) == normalize(s)


def test_rejects_non_string():
    with pytest.raises(TypeError):
        normalize(None)
