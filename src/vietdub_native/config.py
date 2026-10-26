"""Checkpointed hyperparameters and voice/emotion vocabularies."""
from __future__ import annotations
from dataclasses import dataclass, asdict

EMOTIONS = ("neutral", "happy", "sad", "angry", "fearful", "crying", "whispering", "surprised")


@dataclass(frozen=True)
class Config:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    hidden: int = 128
    encoder_layers: int = 3
    decoder_layers: int = 3
    max_frames: int = 3000

    def to_dict(self):
        return asdict(self)
