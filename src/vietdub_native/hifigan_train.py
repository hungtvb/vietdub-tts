"""Independent rights-gated HiFi-GAN-inspired vocoder training/evaluation.

Training requires verified local corpus approvals. This cannot synthesize text;
reconstruction compares waveform fidelity before acoustic-model integration.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import Config
from .audio import mel_spectrogram, read_audio
from .hifigan import HiFiGANGenerator, JointDiscriminator, discriminator_loss, generator_loss, mel_reconstruction_loss
from .vocoder import spectral_loss
from .vocoder_corpus import CorpusError, VocoderWaveDataset


def train_hifigan(manifest: str | Path, output_dir: str | Path, *,
                  cfg: Config | None = None, steps: int = 1000,
                  crop_frames: int = 48, generator_channels: int = 128,
                  discriminator_channels: int = 16, lr: float = 2e-4,
                  device: str = 'cpu', seed: int = 7, resume: bool = False,
                  checkpoint_every: int = 250, eval_seconds: float = 5.0) -> dict:
    """Train G with joint period/scale critics; evaluate a held-out recording."""
    if steps < 1 or lr <= 0 or checkpoint_every < 1 or not 0.5 <= eval_seconds <= 15:
        raise ValueError('steps and learning rate must be positive')
    if crop_frames < 8:
        raise ValueError('Need at least 8 mel frames per training crop')
    cfg = cfg or Config()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Output directory is not empty; refuse to overwrite experiments')
    train = VocoderWaveDataset(manifest, cfg=cfg, split='train', crop_frames=crop_frames, seed=seed)
    dev = VocoderWaveDataset(manifest, cfg=cfg, split='dev', crop_frames=crop_frames, seed=seed)
    if len(train) < 2:
        raise CorpusError('At least two independent approved train clips required')
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    gen = HiFiGANGenerator(cfg, channels=generator_channels).to(device)
    disc = JointDiscriminator(channels=discriminator_channels).to(device)
    gen_opt = torch.optim.AdamW(gen.parameters(), lr=lr, betas=(.8, .99))
    disc_opt = torch.optim.AdamW(disc.parameters(), lr=lr, betas=(.8, .99))
    # Bind resume to actual prepared audio bytes AND provenance metadata.
    manifest_bytes = Path(manifest).read_bytes()
    inputs = [f'{r["split"]}:{r["prepared_sha256"]}' for r in train.rows + dev.rows]
    corpus_digest = hashlib.sha256(manifest_bytes + '\n'.join(inputs).encode()).hexdigest()
    expected = dict(config=asdict(cfg), manifest_sha256=corpus_digest,
                    generator_channels=generator_channels,
                    discriminator_channels=discriminator_channels,
                    crop_frames=crop_frames, lr=lr, seed=seed)
    checkpoint_dir = output / 'private_checkpoints'
    state_path = checkpoint_dir / 'training_state.pt'
    losses = []
    start_step = 0
    if resume:
        if not state_path.is_file():
            raise ValueError('Cannot resume: training state checkpoint missing')
        state = torch.load(state_path, map_location=device, weights_only=True)
        if state.get('type') != 'vietdub_hifigan_training_v3':
            raise ValueError('Cannot resume: checkpoint schema version changed (v3 joint critics); start a new private output directory')
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError('Cannot resume: corpus manifest or configuration changed')
        start_step = state['step']
        if steps <= start_step:
            raise ValueError('Resume target steps must exceed checkpoint step')
        gen.load_state_dict(state['generator_weights'])
        disc.load_state_dict(state['discriminator_weights'])
        gen_opt.load_state_dict(state['generator_optimizer'])
        disc_opt.load_state_dict(state['discriminator_optimizer'])
        losses = state['losses']
        if len(losses) != start_step:
            raise ValueError('Resume checkpoint history mismatch')
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if device.startswith('cuda') and 'cuda_rng_state' in state:
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])

    def persist(step_number: int) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {'type': 'vietdub_hifigan_training_v3', **expected,
                 'step': step_number, 'generator_weights': gen.state_dict(),
                 'discriminator_weights': disc.state_dict(),
                 'generator_optimizer': gen_opt.state_dict(),
                 'discriminator_optimizer': disc_opt.state_dict(),
                 'torch_rng_state': torch.get_rng_state(), 'losses': losses}
        if device.startswith('cuda'):
            state['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        pending = state_path.with_suffix('.pt.part')
        torch.save(state, pending)
        os.replace(pending, state_path)

    output.mkdir(parents=True, exist_ok=True)
    for step in range(start_step, steps):
        mels, original = train.sample_at(step % len(train), step=step)
        mels = mels.unsqueeze(0).to(device)
        truth = original.unsqueeze(0).to(device)
        predicted = gen(mels)
        if predicted.shape != truth.shape:
            raise RuntimeError('Vocoder mel/frame mismatch')
        real_scores, _ = disc(truth)
        fake_scores, _ = disc(predicted.detach())
        d_loss = discriminator_loss(real_scores, fake_scores)
        if not torch.isfinite(d_loss):
            raise RuntimeError('Non-finite discriminator loss')
        disc_opt.zero_grad(set_to_none=True)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.)
        disc_opt.step()

        for parameter in disc.parameters():
            parameter.requires_grad_(False)
        fake_scores, fake_features = disc(predicted)
        with torch.no_grad():
            _, real_features = disc(truth)
        stft = spectral_loss(predicted, truth)
        mel_loss = mel_reconstruction_loss(predicted, truth, cfg)
        g_loss = (generator_loss(fake_scores, fake_features, real_features)
                  + 4.0 * stft + 10.0 * mel_loss
                  + 0.5 * (predicted - truth).abs().mean())
        if not torch.isfinite(g_loss):
            raise RuntimeError('Non-finite generator loss')
        gen_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.)
        gen_opt.step()
        for parameter in disc.parameters():
            parameter.requires_grad_(True)
        losses.append({'step': step + 1, 'generator_loss': round(float(g_loss.detach()), 5),
                       'discriminator_loss': round(float(d_loss.detach()), 5),
                       'mel_reconstruction_loss': round(float(mel_loss.detach()), 5)})
        if (step + 1) % checkpoint_every == 0:
            persist(step + 1)

    persist(steps)
    # Keep legacy inference-only artifact, but do not discard optimizer/critic state.
    torch.save({'type': 'vietdub_hifigan_research_v1', 'config': asdict(cfg),
                'generator_channels': generator_channels, 'step': steps,
                'weights': {k: v.detach().cpu() for k, v in gen.state_dict().items()}},
               checkpoint_dir / 'hifigan.pt')

    # Full held-out dev recording (up to eval_seconds), never used in training.
    # Preserve duration instead of producing an unlistenable 0.2s crop.
    dev_wave = read_audio(dev.rows[0]['resolved_audio'], cfg.sample_rate)
    max_samples = min(int(eval_seconds * cfg.sample_rate), len(dev_wave))
    frames = max_samples // cfg.hop_length
    if frames < 8:
        raise RuntimeError('Held-out recording too short for evaluation')
    dev_wave = dev_wave[:frames * cfg.hop_length]
    dev_mel = mel_spectrogram(dev_wave, cfg)[:frames].T.contiguous()
    gen.eval()
    with torch.inference_mode():
        reconstructed = gen(dev_mel.unsqueeze(0).to(device)).flatten().cpu().numpy()
    reference = dev_wave.numpy()
    if not np.isfinite(reconstructed).all():
        raise RuntimeError('Non-finite validation output')
    sf.write(output / 'dev_reconstructed.wav', reconstructed, cfg.sample_rate, subtype='PCM_16')
    with torch.inference_mode():
        dev_logmel_l1 = float(mel_reconstruction_loss(
            torch.from_numpy(reconstructed).unsqueeze(0),
            torch.from_numpy(reference.copy()).unsqueeze(0), cfg))
    with torch.inference_mode():
        silence_logmel_l1 = float(mel_reconstruction_loss(
            torch.zeros((1, len(reference)), dtype=torch.float32),
            torch.from_numpy(reference.copy()).unsqueeze(0), cfg))
    ac_rms = float(np.std(reconstructed))
    dc_ratio = float(abs(np.mean(reconstructed)) / max(np.sqrt(np.mean(reconstructed ** 2)), 1e-10))
    report = {'status': 'RECONSTRUCTION_REQUIRES_LISTENING_REVIEW',
              'intelligibility_verified': False, 'production_approved': False,
              'note': 'Synthetic CI fixtures are not evidence of intelligible Vietnamese speech.',
              'train_clips': len(train), 'dev_clips': len(dev), 'steps': steps,
              'train_generator_loss_last': losses[-1]['generator_loss'],
              'train_discriminator_loss_last': losses[-1]['discriminator_loss'],
              'dev_reconstruction_mae': round(float(np.mean(abs(reconstructed - reference))), 6),
              'dev_logmel_l1': round(dev_logmel_l1, 6),
              'dev_silence_logmel_l1': round(silence_logmel_l1, 6),
              'dev_logmel_l1_ratio_to_silence': round(dev_logmel_l1 / max(silence_logmel_l1, 1e-7), 6),
              'dev_original_rms': round(float(np.sqrt(np.mean(reference ** 2))), 6),
              'dev_clipped_fraction': round(float(np.mean(np.abs(reconstructed) >= 0.999)), 6),
              'discriminator_architecture': 'joint_period_scale_v1',
              'training_checkpoint_format': 'vietdub_hifigan_training_v3',
              'dev_reconstruction_duration_sec': round(len(reconstructed) / cfg.sample_rate, 3),
              'resume_from_step': start_step,
              'corpus_manifest_sha256': corpus_digest,
              'dev_reconstruction_ac_rms': round(ac_rms, 6),
              'dev_reconstruction_dc_ratio': round(dc_ratio, 6),
              'sample_rate': cfg.sample_rate,
              'warning': 'Source licenses, voice rights and subjective quality must be independently checked.'}
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (output / 'training_history.json').write_text(json.dumps(losses, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Native HiFi-GAN-inspired vocoder trainer')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--crop-frames', type=int, default=48)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--eval-seconds', type=float, default=5.)
    args = parser.parse_args(argv)
    print(json.dumps(train_hifigan(args.manifest, args.output_dir, steps=args.steps,
                                   crop_frames=args.crop_frames, device=args.device, resume=args.resume,
                                   checkpoint_every=args.checkpoint_every,
                                   eval_seconds=args.eval_seconds), indent=2))


if __name__ == '__main__':
    main()
