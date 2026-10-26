import json
import numpy as np
import pytest
import soundfile as sf
from vietdub_native.config import Config
from vietdub_native.audio import mel_spectrogram
from vietdub_native.symbols import encode
from vietdub_native.dataset import AlignedSpeechDataset
from vietdub_native.train import seed_all


@pytest.fixture
def tiny_dataset(tmp_path):
    cfg = Config(n_fft=256, n_mels=16, hidden=32, encoder_layers=1, decoder_layers=1, max_frames=300)
    sr = cfg.sample_rate
    t = np.arange(int(sr * 0.36)) / sr
    audio = (0.08 * np.sin(2 * np.pi * 240 * t)).astype("float32")
    sf.write(tmp_path / "voice.wav", audio, sr)
    import torch
    num_frames = len(mel_spectrogram(torch.from_numpy(audio), cfg))
    text = "xin chào"
    n = len(encode(text))
    durations = [num_frames // n + (i < num_frames % n) for i in range(n)]
    row = {"text": text, "audio_path": "voice.wav", "speaker_id": "actor_a", "emotion": "sad",
           "emotion_intensity": .75, "token_durations": durations}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    seed_all(3)
    return tmp_path / "manifest.jsonl", cfg
