"""Explicitly aligned Vietnamese speech dataset; no synthetic/fake alignment.

Every item must provide `token_durations` in mel frames for each normalized
Unicode grapheme, including spaces and punctuation. This v0 research contract
assumes alignments have already been verified by the dataset producer.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .audio import read_audio, mel_spectrogram
from .config import Config, EMOTIONS
from .symbols import encode, unknown_characters, PAD, PHONETIC_MODE


class ManifestError(ValueError):
    pass


def read_manifest(path: str | Path, tokenizer: str = "grapheme") -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("Expected an object")
            for k in ("text", "audio_path", "speaker_id", "emotion", "token_durations"):
                if k not in row:
                    raise ValueError(f"Missing field {k}")
            if row.get("tokenizer", "grapheme") != tokenizer:
                raise ValueError(f"Alignment tokenizer mismatch: expected {tokenizer}")
            if tokenizer == PHONETIC_MODE and (
                row.get("alignment_verified") is not True
                or not str(row.get("alignment_method", "")).strip()
            ):
                raise ValueError("Phonetic alignment requires reviewed method and alignment_verified=true")
            if unknown_characters(row["text"], tokenizer):
                raise ValueError(
                    f"Unrecognized tokens: {unknown_characters(row['text'], tokenizer)}"
                )
            tokens = encode(row["text"], tokenizer)
            d = row["token_durations"]
            if len(d) != len(tokens) or any(type(n) is not int or n < 0 for n in d) or sum(d) < 1:
                raise ValueError("token_durations must be nonnegative integer frames, 1 per grapheme")
            if row["emotion"] not in EMOTIONS:
                raise ValueError(f"Emotion must be one of: {EMOTIONS}")
            intensity = row.get("emotion_intensity", 0.5)
            if not isinstance(intensity, (int, float)) or not 0 <= intensity <= 1:
                raise ValueError("emotion_intensity must be 0..1")
            if not isinstance(row["speaker_id"], str) or not row["speaker_id"].strip():
                raise ValueError("speaker_id required")
            audio = (path.parent / row["audio_path"]).resolve()
            if not audio.is_file():
                raise ValueError(f"Missing audio file: {audio}")
            result.append({**row, "audio_path": str(audio), "tokens": tokens})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            raise ManifestError(f"{path}:{line_no}: {e}") from e
    if not result:
        raise ManifestError("Manifest is empty")
    return result


class AlignedSpeechDataset(Dataset):
    def __init__(self, manifest: str | Path, cfg: Config, speakers: list[str]):
        self.rows = read_manifest(manifest, tokenizer=cfg.tokenizer)
        self.cfg = cfg
        self.speakers = {name: i for i, name in enumerate(speakers)}
        for row in self.rows:
            if row["speaker_id"] not in self.speakers:
                raise ManifestError(f"Unregistered speaker: {row['speaker_id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wav = read_audio(row["audio_path"], self.cfg.sample_rate)
        if wav.numel() < self.cfg.n_fft:
            raise ManifestError(f"Short recording: {row['audio_path']}")
        mel = mel_spectrogram(wav, self.cfg)
        durations = row["token_durations"]
        if mel.shape[0] != sum(durations):
            raise ManifestError(
                f"{row['audio_path']}: aligned frames {sum(durations)} != measured mel frames {mel.shape[0]}; "
                "redo alignment with the configured sample rate and hop length"
            )
        if mel.shape[0] > self.cfg.max_frames:
            raise ManifestError("Clip exceeds max_frames; split long dialogue into segments")
        return {
            "tokens": torch.tensor(row["tokens"], dtype=torch.long),
            "durations": torch.tensor(durations, dtype=torch.long),
            "mel": mel,
            "wave": wav,
            "speaker": self.speakers[row["speaker_id"]],
            "emotion": EMOTIONS.index(row["emotion"]),
            "intensity": float(row.get("emotion_intensity", 0.5)),
        }


def collate_acoustic(items: list[dict]):
    b = len(items)
    max_t = max(len(x["tokens"]) for x in items)
    max_f = max(len(x["mel"]) for x in items)
    m = items[0]["mel"].shape[-1]
    tokens = torch.full((b, max_t), PAD, dtype=torch.long)
    durations = torch.zeros((b, max_t), dtype=torch.long)
    mel = torch.zeros((b, max_f, m))
    for i, item in enumerate(items):
        tokens[i, :len(item["tokens"])] = item["tokens"]
        durations[i, :len(item["tokens"])] = item["durations"]
        mel[i, :len(item["mel"])] = item["mel"]
    return {"tokens": tokens, "durations": durations, "mel": mel,
            "speaker": torch.tensor([x["speaker"] for x in items]),
            "emotion": torch.tensor([x["emotion"] for x in items]),
            "intensity": torch.tensor([x["intensity"] for x in items], dtype=torch.float32)}
