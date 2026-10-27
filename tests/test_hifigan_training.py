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


def test_training_crop_changes_with_step_but_is_reproducible(tmp_path):
    manifest = _corpus(tmp_path)
    cfg = Config(n_fft=256, n_mels=16)
    dataset = VocoderWaveDataset(manifest, cfg, split='train', crop_frames=16, seed=7)
    first_mel, first_audio = dataset.sample_at(0, step=0)
    second_mel, second_audio = dataset.sample_at(0, step=1)
    repeated_mel, repeated_audio = dataset.sample_at(0, step=0)
    assert torch.equal(first_audio, repeated_audio)
    assert torch.equal(first_mel, repeated_mel)
    assert not torch.equal(first_audio, second_audio)


def test_resume_preserves_both_optimizers_and_checkpoint_state(tmp_path):
    manifest = _corpus(tmp_path)
    cfg = Config(n_fft=256, n_mels=16)
    options = dict(cfg=cfg, crop_frames=16, generator_channels=16,
                   discriminator_channels=8, device='cpu', seed=19)
    fresh = train_hifigan(manifest, tmp_path/'fresh', steps=2, **options)
    one_step = train_hifigan(manifest, tmp_path/'resumed', steps=1, **options)
    resumed = train_hifigan(manifest, tmp_path/'resumed', steps=2, resume=True, **options)
    assert fresh['steps'] == resumed['steps'] == 2
    checkpoint_fresh = torch.load(tmp_path/'fresh'/'private_checkpoints'/'training_state.pt', weights_only=True)
    checkpoint_resume = torch.load(tmp_path/'resumed'/'private_checkpoints'/'training_state.pt', weights_only=True)
    assert checkpoint_resume['step'] == 2
    assert checkpoint_resume['discriminator_weights']
    assert checkpoint_resume['discriminator_optimizer']
    assert checkpoint_resume['generator_optimizer']
    for name, tensor in checkpoint_fresh['generator_weights'].items():
        assert torch.allclose(tensor, checkpoint_resume['generator_weights'][name], atol=1e-7), name
    assert one_step['steps'] == 1


def test_resume_rejects_changed_manifest_without_overwriting_checkpoint(tmp_path):
    manifest = _corpus(tmp_path)
    cfg = Config(n_fft=256, n_mels=16)
    options = dict(cfg=cfg, crop_frames=16, generator_channels=16,
                   discriminator_channels=8, device='cpu')
    train_hifigan(manifest, tmp_path/'out', steps=1, **options)
    before = (tmp_path/'out'/'private_checkpoints'/'training_state.pt').read_bytes()
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    rows[0]['license'] = 'changed-license'
    manifest.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises((ValueError, CorpusError), match='manifest|corpus'):
        train_hifigan(manifest, tmp_path/'out', steps=2, resume=True, **options)
    assert before == (tmp_path/'out'/'private_checkpoints'/'training_state.pt').read_bytes()


def test_dev_reconstruction_uses_longer_independent_audio_not_train_crop(tmp_path):
    manifest = _corpus(tmp_path)
    t = np.arange(2 * 22050) / 22050
    sf.write(tmp_path/'2.wav', (0.12*np.sin(2*np.pi*215*t)).astype('float32'), 22050)
    cfg = Config(n_fft=256, n_mels=16)
    result = train_hifigan(manifest, tmp_path/'longdev', cfg=cfg, steps=1,
                           crop_frames=16, generator_channels=16,
                           discriminator_channels=8, device='cpu')
    info = sf.info(tmp_path/'longdev'/'dev_reconstructed.wav')
    assert info.duration >= 1.9
    assert result['dev_reconstruction_duration_sec'] >= 1.9
    assert result['intelligibility_verified'] is False


def test_multi_scale_discriminator_tracks_time_scales_and_generator_gradient():
    from vietdub_native.hifigan import MultiScaleDiscriminator
    msd = MultiScaleDiscriminator(channels=8, scales=3)
    wave = torch.randn(2, 3072, requires_grad=True)
    scores, features = msd(wave)
    assert len(scores) == len(features) == 3
    assert len({tuple(f[0].shape) for f in features}) == 3  # progressively downsampled
    loss = sum(s.square().mean() for s in scores)
    loss.backward()
    assert wave.grad is not None and wave.grad.abs().sum() > 0


def test_mel_reconstruction_loss_is_differentiable_and_exact_for_same_wave():
    from vietdub_native.hifigan import mel_reconstruction_loss
    cfg = Config(n_fft=256, n_mels=16)
    wave = torch.randn(1, 4096).clamp(-.9, .9)
    target = wave.detach()
    identity = mel_reconstruction_loss(wave, target, cfg)
    assert identity.item() < 1e-7
    noisy = (target + .04*torch.randn_like(target)).detach().requires_grad_(True)
    loss = mel_reconstruction_loss(noisy, target, cfg)
    assert loss.item() > 0.001
    loss.backward()
    assert noisy.grad is not None and torch.isfinite(noisy.grad).all()
    assert noisy.grad.abs().sum() > 0


def test_training_reports_both_discriminator_types_and_mel_fidelity(tmp_path):
    manifest = _corpus(tmp_path)
    cfg = Config(n_fft=256, n_mels=16)
    result = train_hifigan(manifest, tmp_path/'v3', cfg=cfg, steps=1,
                           crop_frames=16, generator_channels=16,
                           discriminator_channels=8, device='cpu')
    ckpt = torch.load(tmp_path/'v3'/'private_checkpoints'/'training_state.pt', weights_only=True)
    assert ckpt['type'] == 'vietdub_hifigan_training_v3'
    assert any(k.startswith('period.') for k in ckpt['discriminator_weights'])
    assert any(k.startswith('scale.') for k in ckpt['discriminator_weights'])
    history = json.loads((tmp_path/'v3'/'training_history.json').read_text())
    assert history[0]['mel_reconstruction_loss'] >= 0
    assert history[0]['discriminator_loss'] >= 0
    assert 'dev_logmel_l1' in result
    assert result['dev_logmel_l1'] >= 0
    assert result['intelligibility_verified'] is False


def test_dev_report_compares_to_silent_baseline_without_claiming_intelligibility(tmp_path):
    manifest = _corpus(tmp_path)
    cfg = Config(n_fft=256, n_mels=16)
    report = train_hifigan(manifest, tmp_path/'diagnostic', cfg=cfg, steps=1,
                           crop_frames=16, generator_channels=16,
                           discriminator_channels=8)
    assert report['dev_silence_logmel_l1'] > 0
    assert report['dev_logmel_l1_ratio_to_silence'] >= 0
    assert 0 <= report['dev_clipped_fraction'] <= 1
    assert report['dev_original_rms'] > 0
    assert report['intelligibility_verified'] is False
    assert report['production_approved'] is False


def test_refuses_to_resume_v2_checkpoint_with_v3_model(tmp_path):
    manifest = _corpus(tmp_path)
    cfg = Config(n_fft=256, n_mels=16)
    options = dict(cfg=cfg, steps=1, crop_frames=16, generator_channels=16,
                   discriminator_channels=8)
    out = tmp_path/'versioned'
    train_hifigan(manifest, out, **options)
    path = out/'private_checkpoints'/'training_state.pt'
    state = torch.load(path, weights_only=True)
    state['type'] = 'vietdub_hifigan_training_v2'
    torch.save(state, path)
    with pytest.raises(ValueError, match='version|schema|checkpoint'):
        train_hifigan(manifest, out, resume=True, **{**options, 'steps': 2})
