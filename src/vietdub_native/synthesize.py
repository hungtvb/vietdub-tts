"""Only permits speech inference from explicit, locally trained checkpoints."""
from __future__ import annotations
from pathlib import Path
import torch
from .audio import save_wav
from .checkpoint import load_acoustic, load_vocoder
from .symbols import encode, unknown_characters


class NativeSynthesizer:
    def __init__(self, acoustic_ckpt: str | Path, vocoder_ckpt: str | Path, device="cpu"):
        self.device = torch.device(device)
        self.acoustic = load_acoustic(acoustic_ckpt, device)
        self.vocoder = load_vocoder(vocoder_ckpt, device)
        if self.acoustic.cfg != self.vocoder.cfg:
            raise ValueError("Acoustic and vocoder checkpoint audio configurations differ")

    @torch.inference_mode()
    def speak(self, text: str, speaker_id: str, emotion: str = "neutral",
              intensity: float = 0.5, out_path: str | Path = "output.wav",
              duration_scale: float = 1.0):
        if speaker_id not in self.acoustic.speakers:
            raise ValueError(f"Unknown speaker_id: {speaker_id}; select a trained voice")
        if emotion not in self.acoustic.emotions:
            raise ValueError(f"Unknown emotion: {emotion}")
        if not 0 <= intensity <= 1:
            raise ValueError("Emotion intensity must be 0..1")
        if not 0.8 <= duration_scale <= 1.2:
            raise ValueError("duration_scale limited to 0.8..1.2 for film-dubbing QA")
        if unknown_characters(text):
            raise ValueError(f"Unsupported graphemes: {unknown_characters(text)}")
        ids = torch.tensor([encode(text)], device=self.device)
        sid = torch.tensor([self.acoustic.speakers.index(speaker_id)], device=self.device)
        eid = torch.tensor([self.acoustic.emotions.index(emotion)], device=self.device)
        strength = torch.tensor([intensity], device=self.device)
        # Duration is predicted directly from the acoustic network. To adjust it,
        # regulate predicted lengths BEFORE mel generation rather than post-hoc atempo.
        if duration_scale != 1.0:
            x = self.acoustic.characters(ids)
            cond = (self.acoustic.speaker_embedding(sid) + self.acoustic.emotion_embedding(eid)
                    + self.acoustic.intensity_projection(strength[:, None]))
            x = self.acoustic.encoder(x + cond[:, None, :])
            logdur = self.acoustic.duration_predictor(x.transpose(1, 2)).squeeze(1)
            durations = torch.expm1(logdur.clamp(min=0, max=5)).round().clamp(min=1, max=160)
            durations = (durations * duration_scale).round().long().clamp(min=1)
            mel, _, lengths = self.acoustic(ids, sid, eid, strength, durations)
        else:
            mel, _, lengths = self.acoustic(ids, sid, eid, strength)
        wave = self.vocoder(mel.transpose(1, 2))
        if not torch.isfinite(wave).all():
            raise RuntimeError("Model produced non-finite waveform")
        count = int(lengths[0]) * self.acoustic.cfg.hop_length
        return save_wav(out_path, wave[0, :count], self.acoustic.cfg.sample_rate, output_sr=44100)
