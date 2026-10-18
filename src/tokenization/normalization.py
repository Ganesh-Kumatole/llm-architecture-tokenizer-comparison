"""Text normalization applied identically before EVERY tokenizer (Blueprint §4.2).

Policy norm-v1:
  1. Line endings: CRLF / CR -> LF.
  2. Mojibake repair (default ON). The prepared TinyStories text contains UTF-8
     punctuation that was wrongly decoded as Windows-1252 upstream, e.g. a right single
     quote (U+2019) appears as the three characters 'â€™'. We map those back.
  3. Unicode NFC.
No lowercasing, no stripping, no whitespace collapsing. Files on disk are never
modified: this runs at tokenization time only.
"""
import re
import unicodedata

NORMALIZATION_VERSION = "norm-v1"

# 'â€' (U+00E2 U+20AC) + one cp1252-decoded byte  ->  the punctuation it came from.
_PUNCT_MOJIBAKE = {
    "\u00e2\u20ac\u02dc": "\u2018",  # left single quote
    "\u00e2\u20ac\u2122": "\u2019",  # right single quote / apostrophe
    "\u00e2\u20ac\u0153": "\u201c",  # left double quote
    "\u00e2\u20ac\u009d": "\u201d",  # right double quote (only if byte 0x9D survived)
    "\u00e2\u20ac\u201c": "\u2013",  # en dash
    "\u00e2\u20ac\u201d": "\u2014",  # em dash
    "\u00e2\u20ac\u00a6": "\u2026",  # ellipsis
}
# In our data the 0x9D byte of a right double quote was dropped, leaving a bare 'â€'.
# Repair a bare 'â€' only when followed by ASCII, by another 'â', or by end of text;
# anything else is left untouched (and shows up in the residue check).
_PUNCT_RE = re.compile(
    "|".join(re.escape(k) for k in _PUNCT_MOJIBAKE)
    + r"|\u00e2\u20ac(?=$|[\x00-\x7f]|\u00e2)"
)
# Latin-1-range characters double-encoded as 'Ã©' (é), 'Ã±' (ñ), 'Â«' («) ...
_LATIN1_RE = re.compile(r"[\u00c2\u00c3][\u0080-\u00bf]")


def _fix_punct(match):
    return _PUNCT_MOJIBAKE.get(match.group(0), "\u201d")


def _fix_latin1(match):
    s = match.group(0)
    try:
        return s.encode("latin-1").decode("utf-8")
    except UnicodeDecodeError:
        return s


def repair_mojibake(text: str) -> str:
    if text.isascii():          # every pattern involves non-ASCII characters
        return text
    text = _PUNCT_RE.sub(_fix_punct, text)
    return _LATIN1_RE.sub(_fix_latin1, text)


def normalize(text: str, repair: bool = True) -> str:
    if not isinstance(text, str):
        raise TypeError(f"normalize() expects str, got {type(text).__name__}")
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.isascii():          # ASCII is already NFC
        return text
    if repair:
        text = repair_mojibake(text)
    return unicodedata.normalize("NFC", text)


def normalization_config(repair: bool = True) -> dict:
    return {
        "version": NORMALIZATION_VERSION,
        "unicode_form": "NFC",
        "newline": "LF",
        "lowercase": False,
        "strip": False,
        "repair_mojibake": repair,
    }
