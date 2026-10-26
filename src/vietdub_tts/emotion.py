"""Transparent experimental emotion handling; never pretend text tags = acting."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Emotion


@dataclass(frozen=True)
class EmotionPlan:
    text_for_model: str
    control_level: str
    note: str


# The v3 Turbo engine documents these specific Vietnamese non-verbal cues.
# Arbitrary labels such as [crying] or `style="angry"` are NOT supported.
_EXPERIMENTAL = {
    Emotion.laughing: "[cười]",
    Emotion.sighing: "[thở dài]",
}


def plan_emotion(text: str, emotion: Emotion, intensity: float) -> EmotionPlan:
    if emotion == Emotion.neutral:
        return EmotionPlan(text, "neutral", "No explicit emotion conditioning requested")
    cue = _EXPERIMENTAL.get(emotion)
    if cue is not None and intensity >= 0.5:
        return EmotionPlan(f"{cue} {text}", "experimental_cue",
                           "VieNeu v3 cue is experimental; audible outcome is unverified")
    return EmotionPlan(text, "uncontrolled",
                       "Requested emotion has no validated conditioning with this backend; "
                       "use a matching recorded reference or a future fine-tuned model")
