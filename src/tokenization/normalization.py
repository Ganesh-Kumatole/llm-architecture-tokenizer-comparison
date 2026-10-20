"""Text normalization applied identically before EVERY tokenizer.

Policy norm-v1:
  1. Line endings: CRLF / CR -> LF.
  2. Mojibake repair (default ON).
  3. Unicode NFC.
No lowercasing, stripping, or whitespace collapsing.
"""
import re
import unicodedata

NORMALIZATION_VERSION = "norm-v1"

_PUNCT_MOJIBAKE = {
    "\u00e2\u20ac\u02dc": "\u2018",
    "\u00e2\u20ac\u2122": "\u2019",
    "\u00e2\u20ac\u0153": "\u201c",
    "\u00e2\u20ac\u009d": "\u201d",
    "\u00e2\u20ac\u201c": "\u2013",
    "\u00e2\u20ac\u201d": "\u2014",
    "\u00e2\u20ac\u00a6": "\u2026",
}

_PUNCT_RE = re.compile(
    "|".join(re.escape(k) for k in _PUNCT_MOJIBAKE)
    + r"|\u00e2\u20ac(?=$|[\x00-\x7f]|\u00e2)"
)

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
    if text.isascii():
        return text
    text = _PUNCT_RE.sub(_fix_punct, text)
    return _LATIN1_RE.sub(_fix_latin1, text)


def normalize(text: str, repair: bool = True) -> str:
    if not isinstance(text, str):
        raise TypeError(
            f"normalize() expects str, got {type(text).__name__}"
        )

    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    if text.isascii():
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
