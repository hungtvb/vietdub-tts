"""Versioned Vietnamese grapheme tokenizer (baseline; future phonemic G2P)."""
from __future__ import annotations
import unicodedata

# Never reorder this vocabulary after training; checkpoints store the exact list.
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
STOI = {s: i for i, s in enumerate(VOCAB)}
PAD, UNK = 0, 1


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def encode(text: str) -> list[int]:
    norm = normalize(text)
    if not norm:
        raise ValueError("Text must not be empty")
    return [STOI.get(ch, UNK) for ch in norm]


def unknown_characters(text: str) -> list[str]:
    return sorted(set(ch for ch in normalize(text) if ch not in STOI))
