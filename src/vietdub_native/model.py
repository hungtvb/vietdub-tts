"""A native, non-autoregressive emotion-conditioned acoustic TTS model.

The architecture and all weights live here. There is no imported pre-trained
speech synthesis model, external TTS API or downloaded checkpoint.
Training requires explicit grapheme-to-mel-frame durations from alignment.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config, EMOTIONS
from .symbols import VOCAB, PAD


class ConvBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        return self.norm(x + self.dropout(F.gelu(self.conv(x.transpose(1, 2)).transpose(1, 2))))


class AcousticModel(nn.Module):
    def __init__(self, cfg: Config, speakers: list[str], emotions: list[str] | None = None):
        super().__init__()
        if not speakers or len(set(speakers)) != len(speakers):
            raise ValueError("Speakers must be nonempty and unique")
        self.cfg = cfg
        self.speakers = list(speakers)
        self.emotions = list(emotions or EMOTIONS)
        if len(set(self.emotions)) != len(self.emotions):
            raise ValueError("Duplicate emotions")
        h = cfg.hidden
        self.characters = nn.Embedding(len(VOCAB), h, padding_idx=PAD)
        self.speaker_embedding = nn.Embedding(len(self.speakers), h)
        self.emotion_embedding = nn.Embedding(len(self.emotions), h)
        self.intensity_projection = nn.Linear(1, h, bias=False)
        self.encoder = nn.Sequential(*[ConvBlock(h) for _ in range(cfg.encoder_layers)])
        self.duration_predictor = nn.Sequential(nn.Conv1d(h, h, 3, padding=1), nn.ReLU(), nn.Conv1d(h, 1, 1))
        self.decoder = nn.Sequential(*[ConvBlock(h) for _ in range(cfg.decoder_layers)])
        self.mel_projection = nn.Linear(h, cfg.n_mels)

    def forward(self, tokens: torch.Tensor, speaker: torch.Tensor,
                emotion: torch.Tensor, intensity: torch.Tensor,
                durations: torch.Tensor | None = None):
        """Returns padded mel [B, F, M], log-duration [B, T], frame_lengths [B]."""
        if tokens.ndim != 2 or tokens.shape != (speaker.shape[0], tokens.shape[1]):
            raise ValueError("Incorrect tokens batch dimensions")
        if intensity.ndim != 1 or (intensity < 0).any() or (intensity > 1).any():
            raise ValueError("Emotion intensity must be in [0,1]")
        x = self.characters(tokens)
        conditioning = (self.speaker_embedding(speaker) + self.emotion_embedding(emotion)
                        + self.intensity_projection(intensity[:, None]))
        x = self.encoder(x + conditioning[:, None, :])
        log_duration = self.duration_predictor(x.transpose(1, 2)).squeeze(1)
        if durations is None:
            durations = torch.expm1(log_duration.detach().clamp(min=0, max=5)).round().long().clamp(min=1, max=160)
        elif durations.shape != tokens.shape or (durations < 0).any():
            raise ValueError("Invalid teacher-forced durations")
        durations = durations.masked_fill(tokens == PAD, 0)
        lengths = durations.sum(dim=1)
        if torch.any(lengths == 0) or torch.any(lengths > self.cfg.max_frames):
            raise ValueError("Invalid expanded frame length; check alignment or split text")
        expanded = [torch.repeat_interleave(x[i], durations[i], dim=0) for i in range(tokens.shape[0])]
        max_length = max(e.shape[0] for e in expanded)
        padded = x.new_zeros((len(expanded), max_length, x.shape[-1]))
        for i, ex in enumerate(expanded):
            padded[i, :ex.shape[0]] = ex
        mel = self.mel_projection(self.decoder(padded))
        return mel, log_duration, lengths


def acoustic_loss(predicted_mel: torch.Tensor, target_mel: torch.Tensor,
                  lengths: torch.Tensor, predicted_log_dur: torch.Tensor,
                  durations: torch.Tensor, tokens: torch.Tensor):
    if predicted_mel.shape != target_mel.shape:
        raise ValueError(f"Mel shape mismatch: {predicted_mel.shape} != {target_mel.shape}")
    valid_frames = torch.arange(target_mel.shape[1], device=lengths.device)[None, :] < lengths[:, None]
    valid_tokens = tokens != PAD
    mel_l1 = ((predicted_mel - target_mel).abs() * valid_frames[:, :, None]).sum() / (valid_frames.sum() * target_mel.shape[-1]).clamp_min(1)
    duration_mse = (((predicted_log_dur - torch.log1p(durations.float())) ** 2) * valid_tokens).sum() / valid_tokens.sum().clamp_min(1)
    return mel_l1 + 0.1 * duration_mse, {"mel_l1": mel_l1.detach(), "duration_mse": duration_mse.detach()}
