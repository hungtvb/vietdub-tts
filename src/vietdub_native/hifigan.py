"""From-scratch HiFi-GAN-inspired mel-to-wave generator and period critics.

No pretrained checkpoints. Architecture is deliberately independently versioned
from VietDub's original convolutional vocoder.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, padding=d * (kernel_size - 1) // 2, dilation=d)
            for d in (1, 3, 5)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(F.leaky_relu(x, .1))
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(self, cfg: Config, *, channels: int = 128):
        super().__init__()
        if cfg.hop_length != 256:
            raise ValueError('Generator requires 256-sample hop')
        if channels < 16 or channels % 8:
            raise ValueError('channels must be a multiple of 8 and >= 16')
        self.cfg = cfg
        self.channels = channels
        self.pre = nn.Conv1d(cfg.n_mels, channels, 7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        n = channels
        for rate in (8, 8, 4):
            next_n = max(16, n // 2)
            self.ups.append(nn.ConvTranspose1d(n, next_n, rate * 2, stride=rate, padding=rate // 2))
            self.resblocks.append(ResidualBlock(next_n))
            n = next_n
        self.post = nn.Conv1d(n, 1, 7, padding=3)

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        if mels.ndim != 3 or mels.shape[1] != self.cfg.n_mels:
            raise ValueError('Expected mels [batch, n_mels, frames]')
        x = self.pre(mels)
        for up, block in zip(self.ups, self.resblocks):
            x = block(up(F.leaky_relu(x, .1)))
        return torch.tanh(self.post(F.leaky_relu(x, .1))).squeeze(1)


class PeriodCritic(nn.Module):
    def __init__(self, period: int, channels: int = 16):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, channels, (5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(channels, channels * 2, (5, 1), stride=(3, 1), padding=(2, 0)),
            nn.Conv2d(channels * 2, channels * 2, (3, 1), padding=(1, 0)),
        ])
        self.final = nn.Conv2d(channels * 2, 1, (3, 1), padding=(1, 0))

    def forward(self, waveform: torch.Tensor):
        if waveform.ndim != 2:
            raise ValueError('Expected waveform [batch, samples]')
        x = waveform[:, None, :]
        pad = (-x.shape[-1]) % self.period
        if pad:
            x = F.pad(x, (0, pad), mode='replicate')
        x = x.reshape(x.shape[0], 1, x.shape[-1] // self.period, self.period)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), .1)
            features.append(x)
        out = self.final(x)
        return out.flatten(1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, *, channels: int = 16, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.critics = nn.ModuleList([PeriodCritic(p, channels=channels) for p in periods])

    def forward(self, waveform: torch.Tensor):
        pairs = [critic(waveform) for critic in self.critics]
        return [x[0] for x in pairs], [x[1] for x in pairs]


def discriminator_loss(real_scores, fake_scores):
    return sum((1 - r).square().mean() + f.square().mean()
               for r, f in zip(real_scores, fake_scores)) / len(real_scores)


def generator_loss(fake_scores, fake_features, real_features):
    adversarial = sum((1 - f).square().mean() for f in fake_scores) / len(fake_scores)
    matching = sum((a.detach() - b).abs().mean()
                   for real, fake in zip(real_features, fake_features)
                   for a, b in zip(real, fake))
    return adversarial + 2.0 * matching


class ScaleCritic(nn.Module):
    """One waveform-resolution critic; returns intermediate feature maps."""

    def __init__(self, channels: int = 16):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, channels, 15, padding=7),
            nn.Conv1d(channels, channels * 2, 41, stride=4, padding=20, groups=4),
            nn.Conv1d(channels * 2, channels * 4, 41, stride=4, padding=20, groups=4),
            nn.Conv1d(channels * 4, channels * 4, 5, padding=2),
        ])
        self.final = nn.Conv1d(channels * 4, 1, 3, padding=1)

    def forward(self, waveform: torch.Tensor):
        if waveform.ndim != 2:
            raise ValueError('Expected waveform [batch, samples]')
        x = waveform.unsqueeze(1)
        features = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), .1)
            features.append(x)
        return self.final(x).flatten(1), features


class MultiScaleDiscriminator(nn.Module):
    """Multiple waveform resolutions complement the periodic waveform critics."""

    def __init__(self, *, channels: int = 16, scales: int = 3):
        super().__init__()
        if channels < 4 or channels % 4 or not 1 <= scales <= 4:
            raise ValueError('channels must be a multiple of four; scales 1..4')
        self.critics = nn.ModuleList(ScaleCritic(channels) for _ in range(scales))
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, waveform: torch.Tensor):
        if waveform.ndim != 2:
            raise ValueError('Expected waveform [batch, samples]')
        scores, features = [], []
        x = waveform
        for critic in self.critics:
            score, hidden = critic(x)
            scores.append(score)
            features.append(hidden)
            x = self.downsample(x.unsqueeze(1)).squeeze(1)
        return scores, features


class JointDiscriminator(nn.Module):
    """Joint multi-period and multi-scale critics, one optimizer/checkpoint."""

    def __init__(self, *, channels: int = 16):
        super().__init__()
        self.period = MultiPeriodDiscriminator(channels=channels)
        self.scale = MultiScaleDiscriminator(channels=channels)

    def forward(self, waveform: torch.Tensor):
        period_scores, period_features = self.period(waveform)
        scale_scores, scale_features = self.scale(waveform)
        return period_scores + scale_scores, period_features + scale_features


def mel_reconstruction_loss(predicted: torch.Tensor, reference: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Differentiable log-mel fidelity; identical waveforms have zero loss."""
    if predicted.ndim != 2 or predicted.shape != reference.shape:
        raise ValueError('Expected matched [batch, samples] waveform tensors')
    from .audio import mel_spectrogram
    return F.l1_loss(mel_spectrogram(predicted, cfg), mel_spectrogram(reference, cfg))
