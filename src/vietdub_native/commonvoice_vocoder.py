"""Offline Mozilla Common Voice Vietnamese -> rights-gated vocoder manifest.

No auto-download, consent inference, or unverifiable speaker labeling. The
operator must explicitly attest rights and training authorization; defaults
remain review-only and will be blocked by the vocoder's manifest gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import soundfile as sf
import numpy as np

SOURCE_URL = 'https://commonvoice.mozilla.org/'
LICENSE = 'CC0-1.0'


def _ranking(seed: int, key: str) -> str:
    return hashlib.sha256(f'{seed}:{key}'.encode('utf-8')).hexdigest()


def _safe_input(clips_root: Path, name: str) -> Path:
    if not name or Path(name).name != name or name.startswith('.') or '\\' in name:
        raise ValueError(f'unsafe audio path in source TSV: {name!r}')
    candidate = (clips_root / name).resolve()
    if not candidate.is_relative_to(clips_root.resolve()):
        raise ValueError('Source audio path escapes clips directory')
    if not candidate.is_file():
        raise ValueError(f'Missing source clip: {name}')
    return candidate


def prepare_commonvoice(source_root: str | Path, output_dir: str | Path, *,
                        max_clips: int = 10000, seed: int = 7,
                        dev_fraction: float = 0.1, rights_reviewed: bool = False,
                        training_approved: bool = False) -> dict:
    """Decode selected speaker-disjoint clips; produce review-only by default.

    Approvals represent deliberate operator input and are never inferred from
    CC0 metadata or the existence of public audio recordings.
    """
    if not 3 <= max_clips <= 100000:
        raise ValueError('max_clips must be between 3 and 100000')
    if not 0 < dev_fraction < 0.5:
        raise ValueError('dev_fraction must be in (0,0.5)')
    if training_approved and not rights_reviewed:
        raise ValueError('Cannot approve vocoder training before reviewing source rights')
    source = Path(source_root).resolve()
    clips_root = source / 'clips'
    tsv = source / 'validated.tsv'
    output = Path(output_dir).resolve()
    if not tsv.is_file() or not clips_root.is_dir():
        raise ValueError('Expected validated.tsv and clips/ under source root')
    if output == source or output.is_relative_to(source):
        raise ValueError('Output must be outside original source directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty')

    groups: dict[str, list[tuple[str, Path]]] = {}
    skipped_missing_client = 0
    seen_names: set[str] = set()
    speaker_by_name: dict[str, str] = {}
    with tsv.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not {'client_id', 'path'}.issubset(set(reader.fieldnames or [])):
            raise ValueError('Common Voice validated.tsv requires client_id and path columns')
        for row in reader:
            client = (row.get('client_id') or '').strip()
            name = (row.get('path') or '').strip()
            if not client:
                skipped_missing_client += 1
                continue  # Speaker-disjoint holdout cannot be guaranteed.
            audio = _safe_input(clips_root, name)  # Validate *before* selection.
            if name in seen_names:
                continue
            seen_names.add(name)
            speaker_by_name[name] = client
            groups.setdefault(client, []).append((name, audio))
    if len(groups) < 3:
        raise ValueError('Need at least three distinct labeled speakers for speaker-disjoint train/dev')

    speakers = sorted(groups, key=lambda s: _ranking(seed, s))
    dev_count = min(len(speakers) - 2, max(1, round(len(speakers) * dev_fraction)))
    dev_ids = set(speakers[:dev_count])
    train_pool = [item for speaker in speakers if speaker not in dev_ids
                  for item in groups[speaker]]
    dev_pool = [item for speaker in speakers if speaker in dev_ids
                for item in groups[speaker]]
    train_pool.sort(key=lambda item: _ranking(seed, 'train:' + item[0]))
    dev_pool.sort(key=lambda item: _ranking(seed, 'dev:' + item[0]))
    if len(train_pool) < 2 or not dev_pool:
        raise ValueError('Insufficient train and dev recordings after speaker split')
    # Reserve samples from two different train speakers plus a held-out
    # speaker before filling the remaining bounded import; transcript/row
    # order or a speaker's clip volume must not defeat the separation.
    train_speakers = [s for s in speakers if s not in dev_ids]
    reserved = [('train', min(groups[s], key=lambda item: _ranking(seed, item[0])))
                for s in train_speakers[:2]]
    reserved.append(('dev', dev_pool[0]))
    selected_names = {item[0] for _, item in reserved}
    rest = ([('train', row) for row in train_pool if row[0] not in selected_names]
            + [('dev', row) for row in dev_pool if row[0] not in selected_names])
    rest.sort(key=lambda pair: _ranking(seed, pair[0] + ':' + pair[1][0]))
    ordered = reserved + rest

    out_audio = output / 'audio'
    out_audio.mkdir(parents=True)
    rows = []
    hashes: set[str] = set()
    rejected_low_quality = 0
    rejected_decode = 0
    for split, (source_path, input_path) in ordered:
        if len(rows) >= max_clips:
            break
        raw_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if raw_hash in hashes:
            continue
        target_name = f'{raw_hash[:24]}.wav'
        target = out_audio / target_name
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
             '-i', str(input_path), '-vn', '-ac', '1', '-ar', '22050',
             '-c:a', 'pcm_s16le', str(target)],
            capture_output=True, text=True, timeout=40, check=False,
        )
        if result.returncode:
            rejected_decode += 1
            target.unlink(missing_ok=True)
            continue
        try:
            info = sf.info(str(target))
            if info.channels != 1 or info.samplerate != 22050 or not 0.4 <= info.duration <= 20:
                rejected_low_quality += 1
                target.unlink(missing_ok=True)
                continue
            samples, _ = sf.read(target, dtype='float32')
            peak = float(np.max(np.abs(samples)))
            rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
            clipped = float(np.mean(np.abs(samples) >= 0.999))
            if not np.isfinite(samples).all() or peak <= 0.001 or rms < 0.002 or clipped > 0.005:
                rejected_low_quality += 1
                target.unlink(missing_ok=True)
                continue
        except (OSError, RuntimeError):
            rejected_decode += 1
            target.unlink(missing_ok=True)
            continue
        hashes.add(raw_hash)
        rows.append({
            'audio_path': 'audio/' + target_name,
            'source_path': source_path,
            'source_sha256': raw_hash,
            'split': split,
            'dataset_source': 'mozilla_common_voice_vi',
            'license': LICENSE,
            'source_url': SOURCE_URL,
            'attribution': 'Mozilla Common Voice Vietnamese; operator must verify release/version',
            'rights_verified': rights_reviewed,
            'vocoder_training_approved': training_approved,
        })
    train_count = sum(row['split'] == 'train' for row in rows)
    dev_count_actual = sum(row['split'] == 'dev' for row in rows)
    if train_count < 2 or not dev_count_actual:
        raise ValueError('Not enough usable train/dev clips after audio validation')
    manifest = output / 'vocoder.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    report = {
        'source': SOURCE_URL, 'license': LICENSE,
        'clips_prepared': len(rows), 'train_clips': train_count,
        'dev_clips': dev_count_actual,
        'train_speakers': len({speaker_by_name[r['source_path']] for r in rows if r['split'] == 'train'}),
        'dev_speakers': len({speaker_by_name[r['source_path']] for r in rows if r['split'] == 'dev'}),
        'skipped_missing_client_id': skipped_missing_client,
        'rejected_low_quality': rejected_low_quality,
        'rejected_decode': rejected_decode,
        'rights_verified_by_operator': rights_reviewed,
        'vocoder_training_approved_by_operator': training_approved,
        'limitations': 'Rights attestation and client_id correctness are not independently verified. No audio quality or speech intelligibility guarantee.',
    }
    (output / 'import_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Offline rights-gated Common Voice Vietnamese vocoder preparation')
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-clips', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--dev-fraction', type=float, default=0.1)
    parser.add_argument('--acknowledge-rights-reviewed', action='store_true')
    parser.add_argument('--approve-vocoder-training', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(prepare_commonvoice(
        args.source_root, args.output_dir, max_clips=args.max_clips,
        seed=args.seed, dev_fraction=args.dev_fraction,
        rights_reviewed=args.acknowledge_rights_reviewed,
        training_approved=args.approve_vocoder_training), indent=2))


if __name__ == '__main__':
    main()
