import unicodedata
from src.tokenization.normalization import normalize

def test_mojibake_repaired():
    broken = "It wasnâ€™t raining."
    fixed = normalize(broken)
    assert "â€™" not in fixed
    assert "’" in fixed

def test_clean_text_idempotent():
    clean = "The fox ran quickly through the forest."
    assert normalize(clean) == clean
    assert normalize(normalize(clean)) == normalize(clean)

def test_nfc_applied():
    decomposed = "cafe\u0301"
    assert normalize(decomposed) == unicodedata.normalize("NFC", decomposed)

def test_newlines_normalized():
    assert normalize("line1\r\nline2\rline3") == "line1\nline2\nline3"

def test_case_and_punctuation_preserved():
    text = "Hello, World! \"Quoted\" text—with an em dash."
    assert normalize(text) == unicodedata.normalize("NFC", text)
