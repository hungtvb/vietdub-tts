"""Stable character-to-voice resolution. References configured by operator only."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class VoiceProfile:
    voice_id: str
    preset: Optional[str] = None
    ref_audio: Optional[str] = None

    def __post_init__(self):
        if bool(self.preset) == bool(self.ref_audio):
            raise ValueError("a voice must configure exactly one preset or ref_audio")


class VoiceRegistry:
    def __init__(self, characters: Optional[Dict[str, VoiceProfile]] = None,
                 default_preset: str = "Hải Đăng"):
        self.characters = dict(characters or {})
        self.default_preset = default_preset

    @classmethod
    def from_json(cls, path: str | Path) -> "VoiceRegistry":
        path = Path(path).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles = {}
        for name, item in data.get("characters", {}).items():
            ref = item.get("ref_audio")
            # Resolve relative references against the CONFIG file, not process CWD.
            if ref:
                reference_path = Path(ref).expanduser()
                ref = str((path.parent / reference_path).resolve()
                          if not reference_path.is_absolute() else reference_path.resolve())
            profiles[name] = VoiceProfile(
                voice_id=name, preset=item.get("preset"), ref_audio=ref)
        return cls(profiles, default_preset=data.get("default_preset", "Hải Đăng"))

    def resolve(self, speaker_id: Optional[str], voice_id: Optional[str]) -> VoiceProfile:
        if speaker_id is not None:
            try:
                profile = self.characters[speaker_id]
            except KeyError as exc:
                raise ValueError(f"unknown speaker_id {speaker_id!r}; configure characters first") from exc
            if voice_id is not None and voice_id not in (speaker_id, profile.preset):
                raise ValueError("voice_id conflicts with registered character voice")
            return profile
        if voice_id:
            # Direct preset selection (no arbitrary reference path in HTTP requests).
            return VoiceProfile(voice_id=voice_id, preset=voice_id)
        return VoiceProfile(voice_id=self.default_preset, preset=self.default_preset)
