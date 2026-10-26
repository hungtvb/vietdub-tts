import numpy as np
import pytest
from vietdub_native.prosody_features import extract_prosody


def test_estimated_pitch_220hz_and_energy():
    sr = 22050
    signal = (0.12*np.sin(2*np.pi*220*np.arange(sr)/sr)).astype('float32')
    stats = extract_prosody(signal)
    assert abs(stats['pitch_median_hz']-220) < 10
    assert stats['voiced_frame_ratio'] > .8
    assert .07 < stats['rms_mean'] < .10
    assert stats['rms_cv'] < .05


def test_silence_has_no_f0():
    stats = extract_prosody(np.zeros(22050, dtype='float32'))
    assert stats['pitch_median_hz'] is None
    assert stats['voiced_frame_ratio'] == 0


def test_short_clip_is_safe():
    stats = extract_prosody(np.ones(100, dtype='float32')*.01)
    assert stats['pitch_median_hz'] is None


def test_invalid_audio_and_settings_rejected():
    with pytest.raises(ValueError):
        extract_prosody(np.array([0., float('nan')]))
    with pytest.raises(ValueError):
        extract_prosody(np.ones(2048),sr=1000)
