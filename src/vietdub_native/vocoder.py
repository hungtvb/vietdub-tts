"""Native trainable mel-to-wave generator (not a pretrained vocoder).

A small transposed-convolution neural vocoder. Initial quality is expected to
be poor. Adversarial training and multi-resolution discriminator are future work.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.leaky_relu(self.c1(F.leaky_relu(x, .2)), .2)) * .2


class NeuralVocoder(nn.Module):
    """Input [B, n_mels, frames], output [B, frames * hop_length]."""
    def __init__(self, cfg: Config, base_channels: int = 128):
        super().__init__()
        if cfg.hop_length != 256:
            raise ValueError("v0 vocoder requires hop_length=256 (8*8*4)")
        self.cfg = cfg
        self.pre = nn.Conv1d(cfg.n_mels, base_channels, 7, padding=3)
        layers = []
        n = base_channels
        for stride in (8, 8, 4):
            next_n = max(16, n // 2)
            layers.append(nn.ConvTranspose1d(n, next_n, stride * 2, stride=stride,
                                             padding=(stride + 1) // 2))
            layers.append(ResidualBlock(next_n))
            n = next_n
        self.layers = nn.ModuleList(layers)
        self.post = nn.Conv1d(n, 1, 7, padding=3)

    def forward(self, mel: torch.Tensor):
        x = self.pre(mel)
        for layer in self.layers:
            x = F.leaky_relu(x, .2) if isinstance(layer, nn.ConvTranspose1d) else x
            x = layer(x)
        return torch.tanh(self.post(F.leaky_relu(x, .2))).squeeze(1)


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-resolution STFT distance; pure tensor gradients to vocoder."""
    losses = []
    for fft, hop in ((256, 64), (512, 128), (1024, 256)):
        window = torch.hann_window(fft, dtype=pred.dtype, device=pred.device)
        a = torch.stft(pred, fft, hop, window=window, return_complex=True).abs()
        b = torch.stft(target, fft, hop, window=window, return_complex=True).abs()
        l = (a - b).abs().mean() + (torch.log1p(a) - torch.log1p(b)).abs().mean()
        losses.append(l)
    return sum(losses) / len(losses)
