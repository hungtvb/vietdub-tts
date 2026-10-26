"""Tests for a new, separately approved and from-scratch neural vocoder."""
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
from vietdub_native.config import Config
from vietdub_native.vocoder_corpus import CorpusError, VocoderWaveDataset, read_vocoder_manifest
from vietdub_native.hifigan import HiFiGANGenerator, MultiPeriodDiscriminator, discriminator_loss, generator_loss
from vietdub_native.hifigan_train import train_hifigan


def _corpus(tmp_path, **changes):
    sr = 22050
    rows = []
    for i, split in enumerate(('train', 'train', 'dev')):
        # Synthetic waveforms are test fixtures only, NEVER claimed to be speech.
        t = np.arange(int(.55 * sr)) / sr
        wave = (0.12 * np.sin(2 * np.pi * (145 + i * 50) * t)).astype('float32')
        sf.write(tmp_path / f'{i}.wav', wave, sr)
        row = dict(audio_path=f'{i}.wav', dataset_source='test_consent_recordings',
                   split=split, rights_verified=True, vocoder_training_approved=True,
                   license='local-test-only', source_url='https://example.org/test',
                   attribution='Synthetic fixture, not human voice',
                   source_sha256=f'{i+1:064x}')
        row.update(changes)
        rows.append(row)
    manifest = tmp_path / 'vocoder.jsonl'
    manifest.write_text(''.join(json.dumps(x) + '\n' for x in rows), encoding='utf-8')
    return manifest


def test_vocoder_manifest_does_not_require_text_alignment_or_emotion(tmp_path):
    manifest = _corpus(tmp_path)
    rows = read_vocoder_manifest(manifest, split='train')
    assert len(rows) == 2
    assert 'token_durations' not in rows[0]
    cfg = Config(n_fft=256, n_mels=16)
    data = VocoderWaveDataset(manifest, cfg=cfg, split='train', crop_frames=24)
    mel, audio = data[0]
    assert mel.shape == (16, 24)
    assert audio.shape == (24 * 256,)
    assert torch.isfinite(mel).all()


def test_vocoder_manifest_rejects_missing_training_approval(tmp_path):
    manifest = _corpus(tmp_path, vocoder_training_approved=False)
    with pytest.raises(CorpusError, match='vocoder_training_approved'):
        read_vocoder_manifest(manifest, split='train')


def test_vocoder_manifest_rejects_path_escape(tmp_path):
    manifest = _corpus(tmp_path, audio_path='../outside.wav')
    with pytest.raises(CorpusError, match='outside'):
        read_vocoder_manifest(manifest, split='train')


def test_vocoder_manifest_rejects_duplicate_across_splits(tmp_path):
    manifest = _corpus(tmp_path)
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    rows[2]['source_sha256'] = rows[0]['source_sha256']
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(CorpusError, match='duplicate'):
        read_vocoder_manifest(manifest, split='train')


def test_generator_output_exact_sample_count_and_backward():
    cfg = Config(n_fft=256, n_mels=16)
    net = HiFiGANGenerator(cfg, channels=32)
    mel = torch.randn(2, 16, 12)
    audio = net(mel)
    assert audio.shape == (2, 12 * cfg.hop_length)
    assert audio.abs().max() <= 1.0
    audio.square().mean().backward()
    assert any(p.grad is not None for p in net.parameters())


def test_multiperoid_discriminator_losses_are_finite_and_differentiable():
    discriminator = MultiPeriodDiscriminator(channels=8, periods=(2, 3))
    real = torch.randn(1, 1024)
    fake = torch.randn(1, 1024, requires_grad=True)
    real_scores, real_features = discriminator(real)
    fake_scores, fake_features = discriminator(fake)
    d_loss = discriminator_loss(real_scores, [s.detach() for s in fake_scores])
    g_loss = generator_loss(fake_scores, fake_features, real_features)
    assert torch.isfinite(d_loss) and torch.isfinite(g_loss)
    g_loss.backward()
    assert fake.grad is not None


def test_one_step_trainer_creates_private_checkpoint_and_dev_metrics(tmp_path):
    manifest = _corpus(tmp_path)
    cfg = Config(n_fft=256, n_mels=16)
    output = tmp_path / 'output'
    result = train_hifigan(manifest, output, cfg=cfg, steps=1,
                           crop_frames=16, generator_channels=16,
                           discriminator_channels=8, device='cpu')
    assert result['steps'] == 1
    assert result['train_clips'] == 2 and result['dev_clips'] == 1
    assert result['intelligibility_verified'] is False
    assert (output / 'private_checkpoints' / 'hifigan.pt').exists()
    assert (output / 'evaluation.json').exists()
    assert (output / 'dev_reconstructed.wav').exists()
    assert not (output / 'dev_original.wav').exists()


def test_manifest_rejects_same_recording_even_with_different_claimed_hashes(tmp_path):
    manifest = _corpus(tmp_path)
    # Copy identical audio bytes but keep different row-declared hashes.
    (tmp_path / '2.wav').write_bytes((tmp_path / '0.wav').read_bytes())
    with pytest.raises(CorpusError, match='duplicate'):
        read_vocoder_manifest(manifest, split='train')


def test_actor_recording_requires_voice_training_consent(tmp_path):
    manifest = _corpus(tmp_path, dataset_source='actor_recordings')
    with pytest.raises(CorpusError, match='consent'):
        read_vocoder_manifest(manifest, split='train')
