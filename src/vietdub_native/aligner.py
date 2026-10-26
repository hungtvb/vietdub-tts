"""Train-from-scratch CTC aligner; alignment suggestions ALWAYS require review."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .audio import read_audio, mel_spectrogram
from .config import Config, EMOTIONS
from .symbols import PAD, encode, unknown_characters, vocabulary


class AlignmentError(ValueError):
    """Input or alignment invalid: review and correct the recordings."""


def _minimal_ctc_frames(tokens: list[int]) -> int:
    return len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))


def read_transcripts(path: str | Path, cfg: Config) -> list[dict]:
    """JSONL speech/transcription rows; token durations not required."""
    manifest = Path(path)
    rows = []
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected a JSON object")
            for field in ("text", "audio_path", "speaker_id", "emotion"):
                if field not in row:
                    raise ValueError(f"Missing field {field}")
            if row.get("tokenizer", cfg.tokenizer) != cfg.tokenizer:
                raise ValueError("Tokenizer mismatch")
            if not isinstance(row["speaker_id"], str) or not row["speaker_id"].strip():
                raise ValueError("speaker_id must be a nonempty string")
            if row["emotion"] not in EMOTIONS:
                raise ValueError("Unknown emotion")
            strength = row.get("emotion_intensity", 0.5)
            if not isinstance(strength, (float, int)) or not 0 <= strength <= 1:
                raise ValueError("Invalid emotion intensity")
            missing = unknown_characters(row["text"], cfg.tokenizer)
            if missing:
                raise ValueError(f"Unsupported tokens: {missing}")
            tokens = encode(row["text"], cfg.tokenizer)
            if PAD in tokens:
                raise ValueError("Padding token cannot be a transcription target")
            audio_path = (manifest.parent / row["audio_path"]).resolve()
            if not audio_path.is_file():
                raise ValueError(f"Missing audio: {audio_path}")
            rows.append({**row, "tokens": tokens, "resolved_audio": str(audio_path)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            raise AlignmentError(f"{manifest}:{line_no}: {e}") from e
    if not rows:
        raise AlignmentError("No transcript rows")
    return rows


class TranscriptDataset(Dataset):
    def __init__(self, manifest: str | Path, cfg: Config):
        self.rows = read_transcripts(manifest, cfg)
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wav = read_audio(row["resolved_audio"], self.cfg.sample_rate)
        if wav.numel() < self.cfg.n_fft:
            raise AlignmentError("Recording shorter than analysis window")
        mel = mel_spectrogram(wav, self.cfg)
        if not 1 <= len(mel) <= self.cfg.max_frames:
            raise AlignmentError("Recording exceeds max_frames")
        tokens = row["tokens"]
        if _minimal_ctc_frames(tokens) > len(mel):
            raise AlignmentError("Not enough mel frames for a CTC transcript; split the dialogue")
        return {"mel": mel, "tokens": torch.tensor(tokens, dtype=torch.long), "row": row}


def collate_transcripts(items: list[dict]) -> dict:
    max_frames = max(len(x["mel"]) for x in items)
    n_mels = items[0]["mel"].shape[-1]
    mels = torch.zeros((len(items), max_frames, n_mels), dtype=torch.float32)
    mel_lengths, token_lengths, flat_tokens = [], [], []
    for i, sample in enumerate(items):
        mels[i, :len(sample["mel"])] = sample["mel"]
        mel_lengths.append(len(sample["mel"]))
        token_lengths.append(len(sample["tokens"]))
        flat_tokens.append(sample["tokens"])
    return {"mels": mels, "mel_lengths": torch.tensor(mel_lengths, dtype=torch.long),
            "token_lengths": torch.tensor(token_lengths, dtype=torch.long),
            "tokens": torch.cat(flat_tokens)}


class CTCAligner(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden
        self.frontend = nn.Sequential(
            nn.Conv1d(cfg.n_mels, hidden, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
        )
        self.encoder = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden * 2, len(vocabulary(cfg.tokenizer)))

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        if mels.ndim != 3 or mels.shape[-1] != self.cfg.n_mels:
            raise ValueError("Expected mel batch [batch, frames, n_mels]")
        features = self.frontend(mels.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.encoder(features)
        return F.log_softmax(self.classifier(encoded), dim=-1)


def ctc_loss(model: CTCAligner, batch: dict) -> torch.Tensor:
    log_prob = model(batch["mels"]).transpose(0, 1)
    loss = F.ctc_loss(log_prob, batch["tokens"], batch["mel_lengths"],
                      batch["token_lengths"], blank=PAD, zero_infinity=False)
    if not torch.isfinite(loss):
        raise AlignmentError("CTC loss is not finite; inspect transcript/audio lengths")
    return loss


def train_aligner(manifest: str | Path, out_path: str | Path, *, cfg: Config | None = None,
                  epochs: int = 1, batch_size: int = 2, lr: float = 2e-4,
                  device: str = "cpu", seed: int = 7) -> list[dict]:
    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    cfg = cfg or Config()
    torch.manual_seed(seed)
    ds = TranscriptDataset(manifest, cfg)
    if any("dataset_source" in row for row in ds.rows):
        from .data_audit import audit_manifests
        report = audit_manifests([manifest], stage="aligner")
        stats = report["by_split"]["train"]
        if stats["total"] == 0 or stats["ready"] != stats["total"]:
            raise AlignmentError(f"Real corpus not approved for CTC training: {report['failure_reasons']}")
    model = CTCAligner(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batches = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_transcripts)
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        for batch in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = ctc_loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "ctc_loss": sum(losses) / len(losses)})
    save_aligner(out_path, model, epochs)
    return history


def save_aligner(path: str | Path, model: CTCAligner, epoch: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"type": "vietdub_ctc_aligner_v1", "config": asdict(model.cfg),
                "vocab": vocabulary(model.cfg.tokenizer), "epoch": epoch,
                "weights": model.state_dict()}, path)


def load_aligner(path: str | Path, device: str = "cpu") -> CTCAligner:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("type") != "vietdub_ctc_aligner_v1":
        raise AlignmentError("Not a VietDub native CTC aligner")
    cfg = Config(**checkpoint["config"])
    if checkpoint.get("vocab") != vocabulary(cfg.tokenizer):
        raise AlignmentError("Aligner tokenizer or vocabulary mismatch")
    model = CTCAligner(cfg)
    model.load_state_dict(checkpoint["weights"])
    return model.to(device).eval()


def ctc_viterbi_durations(log_probs: np.ndarray | torch.Tensor, tokens: list[int],
                          blank: int = PAD) -> tuple[list[int], float]:
    """Monotonic CTC Viterbi path; assign blank frames to nearest token."""
    probs = (log_probs.detach().cpu().numpy() if isinstance(log_probs, torch.Tensor)
             else np.asarray(log_probs))
    if probs.ndim != 2 or not np.isfinite(probs).all():
        raise AlignmentError("Expected finite [frames, vocab] log probabilities")
    frames, classes = probs.shape
    if not tokens or any(type(t) is not int or t <= 0 or t >= classes for t in tokens):
        raise AlignmentError("Invalid transcript token IDs")
    if _minimal_ctc_frames(tokens) > frames:
        raise AlignmentError("CTC needs more frames to align the transcript")
    labels = np.zeros(2 * len(tokens) + 1, dtype=np.int32)
    labels[1::2] = tokens
    states_count = len(labels)
    states = np.full(states_count, -np.inf, dtype=np.float64)
    states[0], states[1] = probs[0, blank], probs[0, tokens[0]]
    trace = np.zeros((frames, states_count), dtype=np.int8)
    skip_ok = np.zeros(states_count, dtype=bool)
    for s in range(3, states_count, 2):
        skip_ok[s] = labels[s] != labels[s-2]
    for t in range(1, frames):
        stay = states
        step = np.r_[-np.inf, states[:-1]]
        skip = np.r_[-np.inf, -np.inf, states[:-2]]
        skip[~skip_ok] = -np.inf
        choices = np.stack((stay, step, skip))
        predecessor = choices.argmax(axis=0)
        trace[t] = predecessor.astype(np.int8)
        states = choices[predecessor, np.arange(states_count)] + probs[t, labels]
    end = states_count-1 if states[-1] >= states[-2] else states_count-2
    if not np.isfinite(states[end]):
        raise AlignmentError("CTC cannot align the transcript")
    path = np.empty(frames, dtype=np.int32)
    state = end
    for t in range(frames-1, -1, -1):
        path[t] = state
        if t:
            state -= int(trace[t, state])
    spans = [np.flatnonzero(path == 2*i + 1) for i in range(len(tokens))]
    if any(not len(x) for x in spans):
        raise AlignmentError("Alignment omitted at least one token")
    boundaries = [0]
    for previous, following in zip(spans, spans[1:]):
        boundaries.append((int(previous[-1]) + int(following[0]) + 1) // 2)
    boundaries.append(frames)
    durations = [b-a for a, b in zip(boundaries, boundaries[1:])]
    if min(durations) <= 0 or sum(durations) != frames:
        raise AlignmentError("Invalid CTC duration assignment")
    posterior = float(np.mean([np.exp(probs[pos, token]).mean()
                               for pos, token in zip(spans, tokens)]))
    return durations, posterior


@torch.inference_mode()
def propose_alignments(input_manifest: str | Path, output_manifest: str | Path,
                       checkpoint: str | Path, *, device: str = "cpu") -> list[dict]:
    """Write unverified token durations; never overwrite the original manifest."""
    src, dest = Path(input_manifest), Path(output_manifest)
    if src.resolve() == dest.resolve():
        raise AlignmentError("Refusing to overwrite input transcription manifest")
    model = load_aligner(checkpoint, device)
    dataset = TranscriptDataset(src, model.cfg)
    checkpoint_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()[:12]
    proposals = []
    for sample in dataset:
        mel = sample["mel"].unsqueeze(0).to(device)
        log_probs = model(mel)[0]
        durations, confidence = ctc_viterbi_durations(log_probs, sample["tokens"].tolist())
        original = {k: v for k, v in sample["row"].items()
                    if k not in ("tokens", "resolved_audio", "token_durations",
                                 "alignment_verified", "alignment_method",
                                 "alignment_status", "alignment_confidence")}
        original["audio_path"] = os.path.relpath(sample["row"]["resolved_audio"], dest.parent.resolve())
        original.update({"tokenizer": model.cfg.tokenizer, "token_durations": durations,
                         "alignment_verified": False,
                         "alignment_status": "needs_human_review",
                         "alignment_method": f"native_ctc_v1:{checkpoint_hash}",
                         "alignment_confidence": round(confidence, 6)})
        proposals.append(original)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".part")
    try:
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                     for row in proposals), encoding="utf-8")
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)
    return proposals
