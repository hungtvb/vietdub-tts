from dataclasses import replace
import json
import numpy as np
import pytest
import soundfile as sf
import torch
from vietdub_native.config import Config, EMOTIONS
from vietdub_native.symbols import encode, normalize, unknown_characters
from vietdub_native.dataset import AlignedSpeechDataset, ManifestError, collate_acoustic, read_manifest
from vietdub_native.model import AcousticModel, acoustic_loss
from vietdub_native.vocoder import NeuralVocoder, spectral_loss
from vietdub_native.checkpoint import save_acoustic, load_acoustic, save_vocoder, load_vocoder
from vietdub_native.train import train_acoustic, train_vocoder
from vietdub_native.synthesize import NativeSynthesizer


def test_vietnamese_tokenization_nfc_and_tone_marks():
    assert normalize("  TIẾNG   Việt \n") == "tiếng việt"
    assert encode("tiếng") == encode("tie\u0302\u0301ng")
    assert encode("ma") != encode("má")
    assert unknown_characters("xin chào") == []
    assert unknown_characters("中文") == ["中", "文"]


def test_aligned_dataset_and_batch(tiny_dataset):
    manifest, cfg = tiny_dataset
    ds = AlignedSpeechDataset(manifest, cfg, ["actor_a"])
    row = ds[0]
    assert len(row["tokens"]) == len(row["durations"])
    assert sum(row["durations"]) == len(row["mel"])
    batch = collate_acoustic([row, row])
    assert batch["tokens"].shape[0] == 2
    assert batch["mel"].shape[2] == cfg.n_mels


def test_rejects_inaccurate_alignment(tiny_dataset):
    manifest, cfg = tiny_dataset
    row = json.loads(manifest.read_text())
    row["token_durations"][0] += 2
    manifest.write_text(json.dumps(row))
    with pytest.raises(ManifestError, match="aligned frames"):
        AlignedSpeechDataset(manifest, cfg, ["actor_a"])[0]


def test_rejects_unlabeled_emotion(tiny_dataset):
    manifest, _ = tiny_dataset
    row = json.loads(manifest.read_text())
    row["emotion"] = "dramatic"
    manifest.write_text(json.dumps(row))
    with pytest.raises(ManifestError, match="Emotion"):
        read_manifest(manifest)


def test_rejects_extra_graphemes(tiny_dataset):
    manifest, _ = tiny_dataset
    row = json.loads(manifest.read_text())
    row["text"] = "你好"
    manifest.write_text(json.dumps(row, ensure_ascii=False))
    with pytest.raises(ManifestError, match="Unrecognized"):
        read_manifest(manifest)


def test_model_forward_backward_and_emotion_condition(tiny_dataset):
    manifest, cfg = tiny_dataset
    sample = AlignedSpeechDataset(manifest, cfg, ["actor_a"])[0]
    batch = collate_acoustic([sample])
    model = AcousticModel(cfg, ["actor_a"])
    pred, logd, lens = model(batch["tokens"], batch["speaker"], batch["emotion"],
                             batch["intensity"], batch["durations"])
    loss, metrics = acoustic_loss(pred, batch["mel"], lens, logd, batch["durations"], batch["tokens"])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.characters.weight.grad is not None
    assert model.emotion_embedding.weight.grad is not None
    angry = torch.tensor([EMOTIONS.index("angry")])
    altered, _, _ = model(batch["tokens"], batch["speaker"], angry, batch["intensity"], batch["durations"])
    assert not torch.allclose(pred, altered)


def test_vocoder_hop_and_gradient(tiny_dataset):
    manifest, cfg = tiny_dataset
    sample = AlignedSpeechDataset(manifest, cfg, ["actor_a"])[0]
    model = NeuralVocoder(cfg, base_channels=32)
    audio = model(sample["mel"].T[None])
    assert audio.shape == (1, len(sample["mel"]) * cfg.hop_length)
    target = torch.nn.functional.pad(sample["wave"], (0, max(0, audio.shape[1]-len(sample["wave"]))))[:audio.shape[1]][None]
    loss = spectral_loss(audio, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.pre.weight.grad is not None


def test_train_and_reload_acoustic(tiny_dataset, tmp_path):
    manifest, cfg = tiny_dataset
    hist = train_acoustic(manifest, tmp_path / "run", cfg=cfg, epochs=1)
    assert len(hist) == 1 and hist[0]["loss"] > 0
    model = load_acoustic(tmp_path / "run/acoustic.pt")
    assert model.speakers == ["actor_a"]
    assert model.cfg == cfg


def test_train_reload_vocoder(tiny_dataset, tmp_path):
    manifest, cfg = tiny_dataset
    hist = train_vocoder(manifest, tmp_path / "run", cfg=cfg, epochs=1)
    assert len(hist) == 1
    assert load_vocoder(tmp_path / "run/vocoder.pt").cfg == cfg


def test_inference_requires_trained_speaker_and_exports_44100(tiny_dataset, tmp_path):
    manifest, cfg = tiny_dataset
    train_acoustic(manifest, tmp_path / "ckpt", cfg=cfg, epochs=1)
    # Checkpoint pair: vocoder below is intentionally UNTRAINED for shape/API smoke test.
    voc = NeuralVocoder(cfg)
    save_vocoder(tmp_path / "ckpt/vocoder.pt", voc)
    synth = NativeSynthesizer(tmp_path / "ckpt/acoustic.pt", tmp_path / "ckpt/vocoder.pt")
    with pytest.raises(ValueError, match="Unknown speaker"):
        synth.speak("xin chào", "someone_else", out_path=tmp_path / "bad.wav")
    with pytest.raises(ValueError, match="duration_scale"):
        synth.speak("xin chào", "actor_a", duration_scale=2.5)
    out = synth.speak("xin chào", "actor_a", "sad", .7, tmp_path / "speech.wav")
    data, sr = sf.read(out)
    assert sr == 44100 and len(data) > 100 and np.isfinite(data).all()


def test_model_rejects_oversized_durations(tiny_dataset):
    manifest, cfg = tiny_dataset
    cfg = replace(cfg, max_frames=5)
    model = AcousticModel(cfg, ["actor_a"])
    ids = torch.tensor([encode("xin chào")])
    with pytest.raises(ValueError, match="expanded frame"):
        model(ids, torch.tensor([0]), torch.tensor([0]), torch.tensor([.5]), torch.ones_like(ids))


def test_checkpoint_config_mismatch(tiny_dataset, tmp_path):
    _, cfg = tiny_dataset
    save_acoustic(tmp_path / "a.pt", AcousticModel(cfg, ["a"]))
    save_vocoder(tmp_path / "v.pt", NeuralVocoder(replace(cfg, n_mels=12)))
    with pytest.raises(ValueError, match="configurations differ"):
        NativeSynthesizer(tmp_path / "a.pt", tmp_path / "v.pt")
