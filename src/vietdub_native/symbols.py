"""Versioned Vietnamese grapheme and experimental phonological tokenizers."""
from __future__ import annotations
import unicodedata
from .text_frontend import MODE as PHONETIC_MODE, tokenize_phonetic, phonetic_vocabulary

_CHARS = (
    " abcdefghijklmnopqrstuvwxyz0123456789.,!?;:'\"-()/+%&@"
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵđ"
)
VOCAB = ["<pad>", "<unk>"] + list(dict.fromkeys(_CHARS))
PHONETIC_VOCAB = phonetic_vocabulary()
PAD, UNK = 0, 1
STOI = {s: i for i, s in enumerate(VOCAB)}
PHONETIC_STOI = {s: i for i, s in enumerate(PHONETIC_VOCAB)}
TOKENIZERS = ("grapheme", PHONETIC_MODE)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def vocabulary(mode: str = "grapheme") -> list[str]:
    if mode == "grapheme":
        return VOCAB
    if mode == PHONETIC_MODE:
        return PHONETIC_VOCAB
    raise ValueError(f"Unknown tokenizer: {mode}")


def tokenize(text: str, mode: str = "grapheme") -> list[str]:
    if mode == "grapheme":
        norm = normalize(text)
        if not norm:
            raise ValueError("Text must not be empty")
        return list(norm)
    if mode == PHONETIC_MODE:
        return tokenize_phonetic(text)
    raise ValueError(f"Unknown tokenizer: {mode}")


def encode(text: str, mode: str = "grapheme") -> list[int]:
    seq = tokenize(text, mode)
    table = STOI if mode == "grapheme" else PHONETIC_STOI
    if mode == PHONETIC_MODE and any(item not in table for item in seq):
        raise ValueError(f"Unsupported phonological tokens: {[x for x in seq if x not in table]}")
    return [table.get(ch, UNK) for ch in seq]


def unknown_characters(text: str, mode: str = "grapheme") -> list[str]:
    seq = tokenize(text, mode)
    table = STOI if mode == "grapheme" else PHONETIC_STOI
    return sorted(set(ch for ch in seq if ch not in table))
