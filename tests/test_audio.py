import io
import shutil

import numpy as np
import pytest
import soundfile as sf

from vietdub_tts.audio import DurationFitError, render_wav
from vietdub_tts.engine import SynthesizedAudio


def sample(seconds=1.0, rate=48000):
    x = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return SynthesizedAudio(0.1 * np.sin(2 * np.pi * 440 * x), rate)


def test_output_mono_pcm16_at_44100():
    wav, seconds = render_wav(sample())
    data, rate = sf.read(io.BytesIO(wav), dtype="float32")
    info = sf.info(io.BytesIO(wav))
    assert rate == 44100
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert abs(seconds - 1.0) < 0.001
    assert np.max(np.abs(data)) <= 1.0


def test_shorter_audio_fills_slot_with_silence():
    wav, seconds = render_wav(sample(0.5), target_duration=1.2)
    data, rate = sf.read(io.BytesIO(wav), dtype="float32")
    assert seconds == pytest.approx(1.2, abs=1 / rate)
    assert len(data) == round(1.2 * rate)
    assert np.max(np.abs(data[-1000:])) == 0


def test_no_silent_clipping_of_overlong_line():
    with pytest.raises(DurationFitError, match="shorten translation"):
        render_wav(sample(3), target_duration=1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_atempo_fits_modest_overrun():
    wav, seconds = render_wav(sample(1.1), target_duration=1.0)
    data, rate = sf.read(io.BytesIO(wav), dtype="float32")
    assert seconds == pytest.approx(1.0)
    assert len(data) == 44100
    assert np.max(np.abs(data)) > 0.01


def test_invalid_engine_audio_rejected():
    with pytest.raises(ValueError, match="NaN"):
        render_wav(SynthesizedAudio(np.array([np.nan], dtype="float32"), 48000))
