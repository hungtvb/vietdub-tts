"""Deterministic Vietnamese linguistic frontend (experimental phonetic v1).

This is a phonological-feature tokenizer, NOT a validated IPA transcriber or
forced aligner. Multi-letter consonant onsets/codas, vowel quality and written
tone are preserved; foreign/ambiguous words fall back to graphemes.
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata

MODE = "vi_phonetic_v1"
TONES = ("ngang", "sac", "huyen", "hoi", "nga", "nang")
_TONE_DIACRITICS = {
    "\u0301": "sac", "\u0300": "huyen", "\u0309": "hoi",
    "\u0303": "nga", "\u0323": "nang",
}
VOWELS = "aăâeêioôơuưy"
ONSETS = (
    "ngh", "ng", "gh", "gi", "qu", "ch", "kh", "nh",
    "ph", "th", "tr", "b", "c", "d", "đ", "g", "h", "k",
    "l", "m", "n", "p", "r", "s", "t", "v", "x",
)
CODAS = ("ng", "nh", "ch", "c", "m", "n", "p", "t", "i", "y", "u", "o")
PUNCTUATION = ".,!?;:-()/'\""
WORD_RE = re.compile(r"[^\W\d_]+|\d+|[^\w\s]", re.UNICODE)
_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")


def read_integer(value: str) -> str:
    """Pronounce integers <= 999,999,999; preserve leading-zero identifiers."""
    if not value.isascii() or not value.isdecimal():
        raise ValueError("Only ASCII decimal digits can be pronounced")
    if len(value) > 1 and value[0] == "0":
        return " ".join(_DIGITS[int(ch)] for ch in value)
    n = int(value)
    if n >= 1_000_000_000:
        raise ValueError("Numbers >= 1 billion need explicit transcript normalization")
    if not n:
        return "không"

    def under_100(k: int, after_hundred: bool = False) -> str:
        if k < 10:
            return ("linh " if after_hundred else "") + _DIGITS[k]
        tens, ones = divmod(k, 10)
        if tens == 1:
            return "mười" + (" " + ("lăm" if ones == 5 else _DIGITS[ones]) if ones else "")
        suffix = ("mốt" if ones == 1 else "tư" if ones == 4 else
                  "lăm" if ones == 5 else _DIGITS[ones]) if ones else ""
        return _DIGITS[tens] + " mươi" + (" " + suffix if suffix else "")

    def group(k: int, full_hundreds: bool) -> str:
        hundreds, rest = divmod(k, 100)
        if hundreds or full_hundreds:
            part = _DIGITS[hundreds] + " trăm"
            return part + (" " + under_100(rest, True) if 0 < rest < 10 else
                           " " + under_100(rest) if rest >= 10 else "")
        return under_100(rest)

    groups = ((n // 1_000_000, "triệu"), ((n // 1_000) % 1000, "nghìn"), (n % 1000, ""))
    result = []
    seen = False
    for k, unit in groups:
        if not k:
            continue
        result.append(group(k, seen and k < 100) + (" " + unit if unit else ""))
        seen = True
    return " ".join(result)


def normalize_for_speech(text: str) -> str:
    """Normalize numbers/dates/times conservatively; retain punctuation."""
    text = unicodedata.normalize("NFC", text).casefold().strip()
    if not text:
        raise ValueError("Text must not be empty")

    def date_sub(m: re.Match) -> str:
        day, month, year = map(int, m.groups())
        try:
            _dt.date(year, month, day)
        except ValueError:
            return m.group(0)
        return (f"ngày {read_integer(str(day))} tháng {read_integer(str(month))} "
                f"năm {read_integer(str(year))}")

    text = re.sub(r"(?<!\w)(\d{1,2})/(\d{1,2})/(\d{4})(?!\w)", date_sub, text)

    def time_sub(m: re.Match) -> str:
        hour, minute = map(int, m.groups())
        if hour > 23 or minute > 59:
            return m.group(0)
        return read_integer(str(hour)) + " giờ" + (
            " " + read_integer(str(minute)) + " phút" if minute else ""
        )

    text = re.sub(r"(?<!\w)(\d{1,2}):(\d{2})(?!\w)", time_sub, text)

    def percentage_sub(m: re.Match) -> str:
        raw = m.group(1)
        if "," in raw or "." in raw:
            left, right = re.split(r"[,.]", raw)
            return read_integer(left) + " phẩy " + " ".join(_DIGITS[int(x)] for x in right) + " phần trăm"
        return read_integer(raw) + " phần trăm"

    text = re.sub(r"(?<!\w)(\d+(?:[,.]\d+)?)%(?!\w)", percentage_sub, text)

    def decimal_sub(m: re.Match) -> str:
        return read_integer(m.group(1)) + " phẩy " + " ".join(_DIGITS[int(x)] for x in m.group(2))

    text = re.sub(r"(?<!\w)(\d+)[,.](\d+)(?!\w)", decimal_sub, text)
    text = re.sub(r"(?<!\w)\d+(?!\w)", lambda m: read_integer(m.group(0)), text)
    return " ".join(text.split())


def _strip_tone(word: str) -> tuple[str, str, int]:
    out = []
    tone = "ngang"
    tone_pos = 0
    for ch in word:
        decomposed = unicodedata.normalize("NFD", ch)
        tone_marks = [x for x in decomposed if x in _TONE_DIACRITICS]
        if tone_marks:
            if tone != "ngang" or len(tone_marks) != 1:
                raise ValueError("Invalid multiple tone marks in a syllable")
            tone = _TONE_DIACRITICS[tone_marks[0]]
            tone_pos = len(out)
        out.append(unicodedata.normalize("NFC", "".join(x for x in decomposed if x not in _TONE_DIACRITICS)))
    return "".join(out), tone, tone_pos


def syllable_tokens(word: str) -> list[str]:
    """Onset + nucleus vowel units (with one tone anchor) + coda.

    Unsupported words are emitted as tagged graphemes instead of inventing
    pronunciations. No automatic forced alignment is performed.
    """
    plain, tone, tone_pos = _strip_tone(word)
    onset = next((x for x in ONSETS if plain.startswith(x)), "")
    body = plain[len(onset):]
    # 'gi' alone still contains a written vowel 'i', not just an onset.
    if onset == "gi" and not body:
        onset, body = "g", "i"
    coda = next(
        (x for x in CODAS if body.endswith(x) and
         len(body) > len(x) and all(c in VOWELS for c in body[:-len(x)])), ""
    )
    nucleus = body[:-len(coda)] if coda else body
    if not nucleus or any(ch not in VOWELS for ch in nucleus):
        return [f"G:{ch}" for ch in word]
    if onset and onset not in ONSETS:
        return [f"G:{ch}" for ch in word]
    result = [f"O:{onset}"] if onset else []
    tone_index = tone_pos - len(onset)
    if tone_index < 0 or tone_index >= len(nucleus):
        tone_index = 0
    result.extend(
        f"V:{v}:{tone if i == tone_index else 'ngang'}"
        for i, v in enumerate(nucleus)
    )
    if coda:
        result.append(f"C:{coda}")
    return result


def tokenize_phonetic(text: str) -> list[str]:
    normalized = normalize_for_speech(text)
    tokens: list[str] = []
    end = 0
    for m in WORD_RE.finditer(normalized):
        if end < m.start() and normalized[end:m.start()].strip() == "":
            if tokens and tokens[-1] != "<space>":
                tokens.append("<space>")
        term = m.group()
        if term.isalpha():
            tokens.extend(syllable_tokens(term))
        elif len(term) == 1 and term in PUNCTUATION:
            tokens.append(f"P:{term}")
        else:
            tokens.extend(f"G:{x}" for x in term)
        end = m.end()
    if not tokens:
        raise ValueError("No speakable tokens")
    return tokens


def phonetic_vocabulary() -> list[str]:
    from itertools import product
    # Stable checkpoint vocabulary. Never reorder a trained checkpoint's IDs.
    fallback = " abcdefghijklmnopqrstuvwxyzđ0123456789"
    return (["<pad>", "<unk>", "<space>"] +
            [f"O:{x}" for x in ONSETS] +
            [f"V:{v}:{t}" for v, t in product(VOWELS, TONES)] +
            [f"C:{x}" for x in CODAS] +
            [f"P:{x}" for x in PUNCTUATION] +
            [f"G:{x}" for x in fallback if x != " "])
