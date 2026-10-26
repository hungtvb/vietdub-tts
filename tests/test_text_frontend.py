"""Versioned Vietnamese phonological frontend and alignment safety tests."""
from dataclasses import replace
import json

import pytest

from vietdub_native.config import Config
from vietdub_native.dataset import AlignedSpeechDataset, ManifestError, read_manifest
from vietdub_native.symbols import (
    VOCAB, PHONETIC_VOCAB, PHONETIC_MODE, TOKENIZERS,
    encode, tokenize, unknown_characters, vocabulary,
)
from vietdub_native.text_frontend import (
    normalize_for_speech, read_integer, syllable_tokens, tokenize_phonetic,
)
from vietdub_native.model import AcousticModel
from vietdub_native.checkpoint import save_acoustic, load_acoustic
from vietdub_native.train import train_acoustic


@pytest.mark.parametrize("written,spoken", [
    ("0", "không"),
    ("15", "mười lăm"),
    ("21", "hai mươi mốt"),
    ("24", "hai mươi tư"),
    ("105", "một trăm linh năm"),
    ("1001", "một nghìn không trăm linh một"),
    ("001", "không không một"),
])
def test_number_pronunciation(written, spoken):
    assert read_integer(written) == spoken


def test_normalizer_handles_date_time_decimal_percentage():
    spoken = normalize_for_speech("Hẹn 12/10/2026, lúc 14:30. Giảm 25,5%!")
    assert "ngày mười hai tháng mười năm hai nghìn không trăm hai mươi sáu" in spoken
    assert "mười bốn giờ ba mươi phút" in spoken
    assert "hai mươi lăm phẩy năm phần trăm" in spoken
    assert spoken.endswith("!")


def test_large_numbers_are_not_silently_converted():
    with pytest.raises(ValueError, match="billion"):
        normalize_for_speech("1000000000")


def test_phonetic_onset_tone_and_coda():
    assert syllable_tokens("nghỉ") == ["O:ngh", "V:i:hoi"]
    assert syllable_tokens("mẹ") == ["O:m", "V:e:nang"]
    assert syllable_tokens("chào") == ["O:ch", "V:a:huyen", "C:o"]
    assert syllable_tokens("quê") == ["O:qu", "V:ê:ngang"]
    assert syllable_tokens("gi") == ["O:g", "V:i:ngang"]
    assert syllable_tokens("học") == ["O:h", "V:o:nang", "C:c"]


def test_phonetic_nfc_whitespace_and_unknown_are_explicit():
    assert encode("tiếng", PHONETIC_MODE) == encode("tie\u0302\u0301ng", PHONETIC_MODE)
    assert "<space>" in tokenize("xin chào", PHONETIC_MODE)
    assert unknown_characters("中文", PHONETIC_MODE) == ["G:中", "G:文"]
    with pytest.raises(ValueError, match="Unsupported"):
        encode("中文", PHONETIC_MODE)
    assert tokenize_phonetic("Tôi vui!")[-1] == "P:!"


def test_versioned_vocab_preserves_existing_grapheme_ids():
    assert vocabulary("grapheme") is VOCAB
    assert vocabulary(PHONETIC_MODE) is PHONETIC_VOCAB
    assert "grapheme" in TOKENIZERS and PHONETIC_MODE in TOKENIZERS
    assert encode("xin chào") == [VOCAB.index(x) for x in "xin chào"]
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        vocabulary("not_supported")


def _phonetic_manifest(tiny_dataset):
    manifest, cfg = tiny_dataset
    row = json.loads(manifest.read_text(encoding="utf-8"))
    n = len(encode(row["text"], PHONETIC_MODE))
    total = sum(row["token_durations"])
    row.update({
        "tokenizer": PHONETIC_MODE,
        "alignment_method": "manual_review_from_recording",
        "alignment_verified": True,
        "token_durations": [total // n + (i < total % n) for i in range(n)],
    })
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest, replace(cfg, tokenizer=PHONETIC_MODE)


def test_phonetic_manifest_requires_alignment_provenance(tiny_dataset):
    manifest, cfg = tiny_dataset
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["tokenizer"] = PHONETIC_MODE
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="alignment_verified"):
        read_manifest(manifest, tokenizer=PHONETIC_MODE)


def test_phonetic_rejects_legacy_grapheme_durations(tiny_dataset):
    manifest, cfg = tiny_dataset
    with pytest.raises(ManifestError, match="tokenizer mismatch"):
        read_manifest(manifest, tokenizer=PHONETIC_MODE)


def test_phonetic_pipeline_train_and_checkpoint_smoke(tiny_dataset, tmp_path):
    # Synthetic durations are shape-only for this test; NOT production alignment.
    manifest, cfg = _phonetic_manifest(tiny_dataset)
    ds = AlignedSpeechDataset(manifest, cfg, ["actor_a"])
    assert len(ds[0]["tokens"]) == len(encode("xin chào", PHONETIC_MODE))
    out = tmp_path / "phonetic"
    history = train_acoustic(manifest, out, cfg=cfg, epochs=1)
    assert history[0]["loss"] >= 0
    model = load_acoustic(out / "acoustic.pt")
    assert model.cfg.tokenizer == PHONETIC_MODE
    assert model.characters.num_embeddings == len(PHONETIC_VOCAB)
    legacy = AcousticModel(replace(cfg, tokenizer="grapheme"), ["actor_a"])
    save_acoustic(tmp_path / "legacy.pt", legacy)
    assert load_acoustic(tmp_path / "legacy.pt").cfg.tokenizer == "grapheme"
