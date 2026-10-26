"""Lazy VieNeu-v3 adapter. Replaceable with an independently trained checkpoint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .voices import VoiceProfile


@dataclass(frozen=True)
class SynthesizedAudio:
    samples: np.ndarray
    sample_rate: int


class SpeechEngine(Protocol):
    def infer(self, text: str, voice: VoiceProfile) -> SynthesizedAudio: ...


class VieNeuEngine:
    """Uses documented VieNeu v3 SDK; model downloads on first model initialization."""

    def __init__(self):
        try:
            from vieneu import Vieneu
        except ImportError as exc:
            raise RuntimeError("Install model dependencies: pip install -e '.[model]'") from exc
        self._model = Vieneu(mode="v3turbo")

    def infer(self, text: str, voice: VoiceProfile) -> SynthesizedAudio:
        if voice.ref_audio:
            # These paths come from the trusted local voice registry only.
            from pathlib import Path
            if not Path(voice.ref_audio).is_file():
                raise ValueError(f"configured voice reference missing: {voice.voice_id}")
            samples = self._model.infer(text, ref_audio=voice.ref_audio)
        else:
            samples = self._model.infer(text, voice=voice.preset)
        return SynthesizedAudio(samples=np.asarray(samples, dtype=np.float32),
                                sample_rate=48000)
