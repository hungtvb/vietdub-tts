"""Native model checkpoints: explicit config and recorded vocabulary."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch
from .config import Config, EMOTIONS
from .model import AcousticModel
from .symbols import VOCAB
from .vocoder import NeuralVocoder


def save_acoustic(path, model: AcousticModel, epoch: int = 0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"type": "vietdub_native_acoustic_v1", "config": asdict(model.cfg),
                "speakers": model.speakers, "emotions": model.emotions,
                "vocab": VOCAB, "epoch": epoch, "weights": model.state_dict()}, path)


def load_acoustic(path, device="cpu") -> AcousticModel:
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["type"] != "vietdub_native_acoustic_v1" or saved["vocab"] != VOCAB:
        raise ValueError("Incompatible acoustic checkpoint and tokenizer")
    model = AcousticModel(Config(**saved["config"]), saved["speakers"], saved["emotions"])
    model.load_state_dict(saved["weights"])
    return model.to(device).eval()


def save_vocoder(path, model: NeuralVocoder, epoch: int = 0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"type": "vietdub_native_vocoder_v1", "config": asdict(model.cfg),
                "epoch": epoch, "weights": model.state_dict()}, path)


def load_vocoder(path, device="cpu") -> NeuralVocoder:
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["type"] != "vietdub_native_vocoder_v1":
        raise ValueError("Not a native vocoder checkpoint")
    model = NeuralVocoder(Config(**saved["config"]))
    model.load_state_dict(saved["weights"])
    return model.to(device).eval()
