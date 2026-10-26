"""Reproducible, from-random-initialization acoustic/vocoder training loops."""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config
from .dataset import AlignedSpeechDataset, collate_acoustic, read_manifest
from .model import AcousticModel, acoustic_loss
from .vocoder import NeuralVocoder, spectral_loss
from .checkpoint import save_acoustic, save_vocoder


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_acoustic(manifest, out_dir, *, epochs=1, batch_size=2, lr=2e-4,
                   cfg=None, device="cpu", seed=7):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    cfg = cfg or Config()
    seed_all(seed)
    speakers = sorted({r["speaker_id"] for r in read_manifest(manifest, tokenizer=cfg.tokenizer)})
    ds = AlignedSpeechDataset(manifest, cfg, speakers)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_acoustic)
    model = AcousticModel(cfg, speakers).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            m, logd, lengths = model(batch["tokens"], batch["speaker"], batch["emotion"], batch["intensity"], batch["durations"])
            loss, _ = acoustic_loss(m, batch["mel"], lengths, logd, batch["durations"], batch["tokens"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite acoustic loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses)})
        save_acoustic(out_dir / "acoustic.pt", model, epoch)
    (out_dir / "acoustic_history.json").write_text(json.dumps(history, indent=2))
    return history


def train_vocoder(manifest, out_dir, *, epochs=1, lr=2e-4,
                  cfg=None, device="cpu", seed=7):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    cfg = cfg or Config()
    seed_all(seed)
    speakers = sorted({r["speaker_id"] for r in read_manifest(manifest)})
    ds = AlignedSpeechDataset(manifest, cfg, speakers)
    model = NeuralVocoder(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        for i in torch.randperm(len(ds)).tolist():
            sample = ds[i]
            mel = sample["mel"].T[None].to(device)
            audio = sample["wave"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(mel)
            samples = pred.shape[-1]
            truth = torch.nn.functional.pad(audio, (0, max(0, samples - len(audio))))[:samples][None]
            loss = spectral_loss(pred, truth) + 0.25 * (pred - truth).abs().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite vocoder loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses)})
        save_vocoder(out_dir / "vocoder.pt", model, epoch)
    (out_dir / "vocoder_history.json").write_text(json.dumps(history, indent=2))
    return history
