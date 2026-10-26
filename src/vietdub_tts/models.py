"""Typed HTTP/SDK contracts. Emotion is metadata until controllability is proven."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Emotion(str, Enum):
    neutral = "neutral"
    joyful = "joyful"
    sad = "sad"
    angry = "angry"
    afraid = "afraid"
    crying = "crying"
    whispering = "whispering"
    sarcastic = "sarcastic"
    excited = "excited"
    laughing = "laughing"
    sighing = "sighing"


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1500)
    speaker_id: Optional[str] = Field(default=None, max_length=80)
    voice_id: Optional[str] = Field(default=None, max_length=120)
    emotion: Emotion = Emotion.neutral
    emotion_intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    target_duration: Optional[float] = Field(default=None, gt=0, le=90)

    @model_validator(mode="after")
    def validate_text(self):
        if not self.text.strip():
            raise ValueError("text must not be whitespace-only")
        if self.speaker_id is not None and not self.speaker_id.strip():
            raise ValueError("speaker_id cannot be empty")
        if self.voice_id is not None and not self.voice_id.strip():
            raise ValueError("voice_id cannot be empty")
        return self


class DubbingSegment(SpeechRequest):
    index: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_timeline(self):
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.target_duration is not None:
            raise ValueError("target_duration is derived from start/end for dubbing")
        if not self.speaker_id and not self.voice_id:
            raise ValueError("every film segment requires speaker_id or voice_id")
        if self.end - self.start > 90:
            raise ValueError("one segment cannot exceed 90 seconds")
        return self


class DubbingRequest(BaseModel):
    segments: List[DubbingSegment] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_indices(self):
        indices = [s.index for s in self.segments]
        if len(set(indices)) != len(indices):
            raise ValueError("segment index must be unique")
        return self
