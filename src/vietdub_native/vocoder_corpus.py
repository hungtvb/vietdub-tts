"""Rights-attested waveform-only corpus for vocoder training.

Unlike the acoustic model, a vocoder needs mel/audio pairs, not transcript,
phoneme durations, emotion or speaker labels. Approval here is an *operator
assertion*, not proof that the asserted permission really exists.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import soundfile as sf
import torch
from torch.utils.data import Dataset

from .audio import mel_spectrogram, read_audio
from .config import Config


class CorpusError(ValueError):
    pass


def read_vocoder_manifest(path: str | Path, *, split: str = 'train') -> list[dict]:
    """Validate the *whole* manifest; detect leakage before selecting a split."""
    if split not in ('train', 'dev'):
        raise CorpusError('split must be train or dev')
    path = Path(path).resolve()
    if not path.is_file():
        raise CorpusError(f'Missing manifest: {path}')
    rows = []
    fingerprints = set()
    for line_number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise CorpusError('Expected JSON object')
            if row.get('split') not in ('train', 'dev'):
                raise CorpusError('Only explicit train/dev split allowed')
            if row.get('rights_verified') is not True:
                raise CorpusError('rights_verified must be explicitly true after human review')
            if row.get('vocoder_training_approved') is not True:
                raise CorpusError('vocoder_training_approved must be explicitly true')
            for field in ('dataset_source', 'license', 'source_url', 'attribution'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise CorpusError(f'Missing source provenance: {field}')
            uri = urlsplit(row['source_url'])
            if uri.scheme not in ('http', 'https') or not uri.netloc:
                raise CorpusError('source_url must be a public provenance URL')
            if row['dataset_source'] == 'google_fleurs_vi':
                raise CorpusError('FLEURS is reserved for evaluation, not vocoder training')
            if row['dataset_source'] == 'actor_recordings' and row.get('voice_training_consent_verified') is not True:
                raise CorpusError('Actor recording requires human-verified voice training consent')
            relative = row.get('audio_path')
            if not isinstance(relative, str) or not relative.strip() or '://' in relative:
                raise CorpusError('audio_path must be a relative local path')
            audio = (path.parent / relative).resolve()
            if not audio.is_relative_to(path.parent):
                raise CorpusError('Audio path outside manifest directory')
            if not audio.is_file():
                raise CorpusError(f'Audio missing: {audio.name}')
            details = sf.info(str(audio))
            if details.samplerate != 22050 or details.channels != 1 or not 0.4 <= details.duration <= 20:
                raise CorpusError('Vocoder requires mono 22050 Hz recordings of 0.4–20 seconds')
            fingerprint = row.get('source_sha256')
            if not (isinstance(fingerprint, str) and len(fingerprint) == 64
                    and all(c in '0123456789abcdef' for c in fingerprint.lower())):
                fingerprint = hashlib.sha256(audio.read_bytes()).hexdigest()
            # Never trust a manifest-provided digest as the only leakage key.
            # Compare the prepared audio bytes as well: different declared SHA
            # values must not bypass train/dev separation.
            prepared_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
            if fingerprint in fingerprints or prepared_hash in fingerprints:
                raise CorpusError('duplicate source audio across train/dev split or within split')
            fingerprints.add(fingerprint)
            fingerprints.add(prepared_hash)
            rows.append({**row, 'resolved_audio': str(audio)})
        except (json.JSONDecodeError, OSError, RuntimeError, CorpusError) as exc:
            raise CorpusError(f'{path.name}:{line_number}: {exc}') from exc
    selected = [row for row in rows if row['split'] == split]
    if not selected:
        raise CorpusError(f'No {split} clips in vocoder corpus')
    return selected


class VocoderWaveDataset(Dataset):
    """Deterministic frame-aligned waveform crops and matching log-mels."""

    def __init__(self, manifest: str | Path, cfg: Config, *, split: str,
                 crop_frames: int, seed: int = 7):
        if crop_frames < 8 or cfg.sample_rate != 22050 or cfg.hop_length != 256:
            raise CorpusError('crop_frames>=8, 22050 Hz and hop_length=256 required')
        self.rows = read_vocoder_manifest(manifest, split=split)
        self.cfg = cfg
        self.crop_frames = crop_frames
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        wave = read_audio(self.rows[index]['resolved_audio'], self.cfg.sample_rate)
        crop_samples = self.crop_frames * self.cfg.hop_length
        if wave.numel() < crop_samples:
            raise CorpusError('Recording is shorter than requested crop')
        max_start_frame = (wave.numel() - crop_samples) // self.cfg.hop_length
        start_frame = (self.seed + index * 7919) % (max_start_frame + 1)
        start = start_frame * self.cfg.hop_length
        audio = wave[start:start + crop_samples]
        mel = mel_spectrogram(audio, self.cfg)[:self.crop_frames].T.contiguous()
        return mel, audio
