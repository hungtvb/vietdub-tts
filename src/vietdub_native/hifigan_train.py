"""Independent rights-gated HiFi-GAN-inspired vocoder training/evaluation.

Training requires verified local corpus approvals. This cannot synthesize text;
reconstruction compares waveform fidelity before acoustic-model integration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import Config
from .audio import mel_spectrogram, read_audio
from .hifigan import HiFiGANGenerator, MultiPeriodDiscriminator, discriminator_loss, generator_loss
from .vocoder import spectral_loss
from .vocoder_corpus import CorpusError, VocoderWaveDataset


def train_hifigan(manifest: str | Path, output_dir: str | Path, *,
                  cfg: Config | None = None, steps: int = 1000,
                  crop_frames: int = 48, generator_channels: int = 128,
                  discriminator_channels: int = 16, lr: float = 2e-4,
                  device: str = 'cpu', seed: int = 7) -> dict:
    """Train G and MPD from random weights; evaluate on a held-out recording."""
    if steps < 1 or lr <= 0:
        raise ValueError('steps and learning rate must be positive')
    if crop_frames < 8:
        raise ValueError('Need at least 8 mel frames per training crop')
    cfg = cfg or Config()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
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
    disc = MultiPeriodDiscriminator(channels=discriminator_channels).to(device)
    gen_opt = torch.optim.AdamW(gen.parameters(), lr=lr, betas=(.8, .99))
    disc_opt = torch.optim.AdamW(disc.parameters(), lr=lr, betas=(.8, .99))
    losses = []
    for step in range(steps):
        mels, original = train[step % len(train)]
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

        fake_scores, fake_features = disc(predicted)
        with torch.no_grad():
            _, real_features = disc(truth)
        stft = spectral_loss(predicted, truth)
        g_loss = (generator_loss(fake_scores, fake_features, real_features)
                  + 4.0 * stft + 0.5 * (predicted - truth).abs().mean())
        if not torch.isfinite(g_loss):
            raise RuntimeError('Non-finite generator loss')
        gen_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.)
        gen_opt.step()
        losses.append({'step': step + 1, 'generator_loss': round(float(g_loss.detach()), 5),
                       'discriminator_loss': round(float(d_loss.detach()), 5)})

    output.mkdir(parents=True, exist_ok=True)
    private = output / 'private_checkpoints'
    private.mkdir()
    torch.save({'type': 'vietdub_hifigan_research_v1', 'config': asdict(cfg),
                'generator_channels': generator_channels, 'step': steps,
                'weights': gen.cpu().state_dict()}, private / 'hifigan.pt')

    # Dev data is never used for an optimizer step. Do not write reference audio.
    dev_mel, dev_wave = dev[0]
    gen.eval()
    with torch.inference_mode():
        reconstructed = gen(dev_mel.unsqueeze(0)).flatten().numpy()
    reference = dev_wave.numpy()
    if not np.isfinite(reconstructed).all():
        raise RuntimeError('Non-finite validation output')
    sf.write(output / 'dev_reconstructed.wav', reconstructed, cfg.sample_rate, subtype='PCM_16')
    ac_rms = float(np.std(reconstructed))
    dc_ratio = float(abs(np.mean(reconstructed)) / max(np.sqrt(np.mean(reconstructed ** 2)), 1e-10))
    report = {'status': 'RECONSTRUCTION_REQUIRES_LISTENING_REVIEW',
              'intelligibility_verified': False, 'production_approved': False,
              'note': 'Synthetic CI fixtures are not evidence of intelligible Vietnamese speech.',
              'train_clips': len(train), 'dev_clips': len(dev), 'steps': steps,
              'train_generator_loss_last': losses[-1]['generator_loss'],
              'train_discriminator_loss_last': losses[-1]['discriminator_loss'],
              'dev_reconstruction_mae': round(float(np.mean(abs(reconstructed - reference))), 6),
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
    args = parser.parse_args(argv)
    print(json.dumps(train_hifigan(args.manifest, args.output_dir, steps=args.steps,
                                   crop_frames=args.crop_frames, device=args.device), indent=2))


if __name__ == '__main__':
    main()
