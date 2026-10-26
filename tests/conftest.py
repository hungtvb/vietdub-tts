import numpy as np
import pytest

from vietdub_tts.engine import SynthesizedAudio
from vietdub_tts.service import TTSService
from vietdub_tts.voices import VoiceProfile, VoiceRegistry


class FakeEngine:
    def __init__(self, seconds=0.5):
        self.calls = []
        self.seconds = seconds

    def infer(self, text, voice):
        self.calls.append((text, voice))
        rate = 48000
        t = np.arange(int(self.seconds * rate), dtype=np.float32) / rate
        return SynthesizedAudio(0.1 * np.sin(2 * np.pi * 220 * t), rate)


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def test_service(fake_engine):
    registry = VoiceRegistry({
        "hero": VoiceProfile("hero", preset="Hải Đăng"),
        "heroine": VoiceProfile("heroine", preset="Mai Anh"),
    })
    return TTSService(engine=fake_engine, voices=registry)
