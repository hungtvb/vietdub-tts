"""Shared audio features. Always use the same parameters in training/inference."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import soundfile as sf
import soxr
import torch
import torchaudio
from .config import Config


def read_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    signal, sr = sf.read(str(path), dtype="float32", always_2d=True)
    signal = signal.mean(axis=1)
    if sr != sample_rate:
        signal = soxr.resample(signal, sr, sample_rate)
    signal = np.asarray(signal, dtype=np.float32)
    if not signal.size or not np.isfinite(signal).all():
        raise ValueError(f"Invalid recording: {path}")
    if max(float(np.max(np.abs(signal))), 0.0) > 1.001:
        raise ValueError(f"Audio exceeds full scale: {path}")
    return torch.from_numpy(signal.copy())


@lru_cache(maxsize=8)
def _mel_transform(sr: int, n_fft: int, hop: int, n_mels: int):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        n_mels=n_mels, power=2.0, center=True, norm="slaney", mel_scale="slaney"
    )


def mel_spectrogram(wave: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Output [frames, n_mels], natural log power mel."""
    transform = _mel_transform(cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels)
    return transform(wave.float()).clamp_min(1e-5).log().transpose(-1, -2)


def save_wav(path: str | Path, wave: torch.Tensor, source_sr: int, output_sr: int = 44100):
    if output_sr <= 0:
        raise ValueError("Invalid sample rate")
    arr = wave.detach().float().cpu().numpy().reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite generated audio")
    if source_sr != output_sr:
        arr = soxr.resample(arr, source_sr, output_sr)
    arr = np.clip(arr, -1, 1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, arr, output_sr, subtype="PCM_16")
    return path
