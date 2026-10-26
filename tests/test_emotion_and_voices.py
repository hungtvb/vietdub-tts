import json

import pytest

from vietdub_tts.emotion import plan_emotion
from vietdub_tts.models import Emotion, SpeechRequest
from vietdub_tts.voices import VoiceRegistry


def test_neutral_no_cue():
    p = plan_emotion("Xin chào", Emotion.neutral, 0.9)
    assert p.text_for_model == "Xin chào"
    assert p.control_level == "neutral"


def test_laugh_experimental_cue_only_when_intense():
    p = plan_emotion("Thật sao!", Emotion.laughing, 0.8)
    assert p.text_for_model.startswith("[cười]")
    assert p.control_level == "experimental_cue"
    p = plan_emotion("Thật sao!", Emotion.laughing, 0.1)
    assert p.text_for_model == "Thật sao!"
    assert p.control_level == "uncontrolled"


def test_angry_does_not_inject_fake_model_tag():
    p = plan_emotion("Dừng lại", Emotion.angry, 1)
    assert p.text_for_model == "Dừng lại"
    assert p.control_level == "uncontrolled"


def test_character_voice_pinned_across_emotions(test_service, fake_engine):
    for emotion in (Emotion.sad, Emotion.angry, Emotion.crying):
        out = test_service.synthesize(SpeechRequest(
            text="Tôi không tin!", speaker_id="hero", emotion=emotion))
        assert out.meta["voice_id"] == "hero"
        assert out.meta["emotion_control"] == "uncontrolled"
    assert [voice.preset for _, voice in fake_engine.calls] == ["Hải Đăng"] * 3


def test_unknown_character_is_not_silently_narrated(test_service):
    with pytest.raises(ValueError, match="unknown speaker_id"):
        test_service.synthesize(SpeechRequest(text="A", speaker_id="unverified_4"))


def test_voice_registry_relative_reference(tmp_path):
    cfg = tmp_path / "voices.json"
    cfg.write_text(json.dumps({"characters": {"lead": {"ref_audio": "assets/my_actor.wav"}}}))
    registry = VoiceRegistry.from_json(cfg)
    assert registry.resolve("lead", None).ref_audio == str((tmp_path / "assets/my_actor.wav").resolve())


def test_whitespace_text_rejected():
    with pytest.raises(Exception):
        SpeechRequest(text="  ")
