"""Offline, bounded preparation of Phitran21 Vietnamese Adam TTS 50h.

Speech is SYNTHETIC, single-preset Adam, not a real consented actor.
Training is blocked by default until the operator separately reviews the
published corpus terms AND upstream synthetic voice / generator rights.
No download, model weights, identity inference or audio redistribution.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import unicodedata
import zipfile

import numpy as np
import soundfile as sf
import soxr

SOURCE_URL = 'https://huggingface.co/datasets/Phitran21/Vietnamese-Adam-TTS-50h'
SOURCE_NAME = 'phitran21_vietnamese_adam_tts_50h'
LICENSE = 'CC-BY-4.0'  # Hugging Face dataset-card label; upstream terms not confirmed.
MAX_COMPRESSED_WAV_BYTES = 6 * 1024 * 1024
MAX_MANIFEST_BYTES = 32 * 1024 * 1024


def _rank(seed: int, key: str) -> str:
    return hashlib.sha256(f'{seed}:{key}'.encode('utf-8')).hexdigest()


def _text_group(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def _validate_zip_archive(zf: zipfile.ZipFile) -> tuple[str, dict[str, zipfile.ZipInfo]]:
    files = {}
    for member in zf.infolist():
        name = member.filename
        parts = PurePosixPath(name).parts
        if (not name or name.startswith('/') or '\\' in name or ':' in name
                or any(part in ('.', '..') for part in parts)):
            raise ValueError(f'unsafe ZIP member: {name!r}')
        if not member.is_dir():
            if name in files:
                raise ValueError(f'Duplicate ZIP member: {name}')
            files[name] = member
    meta = [n for n in files if n.endswith('/data.jsonl') or n == 'data.jsonl']
    if len(meta) != 1:
        raise ValueError('Archive must contain exactly one data.jsonl metadata file')
    return meta[0][: -len('data.jsonl')], files


def _read_metadata(zf: zipfile.ZipFile, entry: zipfile.ZipInfo, *,
                   rejected_transcript_lines: list[int] | None = None) -> list[dict]:
    """Keep structurally valid IDs, quarantine blank/oversized transcripts.

    Skips transcript defects only. Malformed JSON, missing/duplicate IDs and
    corrupt archive metadata remain hard failures. Never truncates text.
    """
    if entry.file_size > MAX_MANIFEST_BYTES:
        raise ValueError('data.jsonl exceeds metadata size limit')
    items = []
    ids = set()
    try:
        raw = zf.read(entry, pwd=None).decode('utf-8-sig')
    except (OSError, RuntimeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise ValueError('Cannot read UTF-8 metadata in Adam TTS ZIP') from exc
    for index, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ident = row['id']
            if type(ident) is not int or ident < 1 or ident in ids:
                raise ValueError('id must be a unique positive integer')
            ids.add(ident)
            text = row.get('text')
            if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
                if rejected_transcript_lines is not None:
                    rejected_transcript_lines.append(index)
                continue
            group = _text_group(text)
            if not group:
                if rejected_transcript_lines is not None:
                    rejected_transcript_lines.append(index)
                continue
            items.append({'id': ident, 'text': text.strip(), 'group': group})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f'data.jsonl line {index}: {exc}') from exc
    if len(items) < 3:
        raise ValueError('Need at least 3 valid transcript rows to split train/dev')
    return items


def prepare_adam_tts(archive_path: str | Path, output_dir: str | Path, *,
                     max_clips: int = 5000, seed: int = 7,
                     dev_fraction: float = 0.05,
                     rights_reviewed: bool = False,
                     upstream_voice_rights_reviewed: bool = False,
                     training_approved: bool = False) -> dict:
    """Read WAV directly from ZIP, resample and emit vocoder-only manifest.

    No extraction of the 17.5 GB archive. Training approvals are assertions
    by the operator after *two separate reviews*, never derived from labels.
    Single preset: split by normalized utterance, NOT speaker (impossible).
    """
    if not 3 <= max_clips <= 50000:
        raise ValueError('max_clips must be 3..50000')
    if not 0 < dev_fraction < 0.5:
        raise ValueError('dev_fraction must be in (0,0.5)')
    if training_approved and not rights_reviewed:
        raise ValueError('Dataset rights review required before training approval')
    if training_approved and not upstream_voice_rights_reviewed:
        raise ValueError('Review upstream Adam voice / generation rights before training approval')

    # A seekable, explicitly bounded HTTP Range reader is also accepted by
    # ZipFile; the remote wrapper owns and closes that stream. Existing local
    # filesystem behaviour and non-overwrite checks remain unchanged.
    streaming = callable(getattr(archive_path, 'read', None)) and callable(getattr(archive_path, 'seek', None))
    output = Path(output_dir).resolve()
    if streaming:
        archive_source = archive_path
    else:
        archive = Path(archive_path).resolve()
        if not archive.is_file() or not zipfile.is_zipfile(archive):
            raise ValueError('Expected a local finnal_tts.zip archive')
        if output == archive.parent or output.is_relative_to(archive):
            raise ValueError('Output must not overwrite source archive')
        archive_source = archive
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty; cannot mix data releases')

    with zipfile.ZipFile(archive_source) as zf:
        prefix, entries = _validate_zip_archive(zf)
        rejected_transcripts: list[int] = []
        records = _read_metadata(zf, entries[prefix + 'data.jsonl'],
                                 rejected_transcript_lines=rejected_transcripts)
        # Select clips deterministically regardless of data.jsonl order.
        records.sort(key=lambda row: _rank(seed, str(row['id'])))
        output.mkdir(parents=True, exist_ok=True)
        (output / 'audio').mkdir()
        rows = []
        rejected_missing = rejected_bad = rejected_duplicate = 0
        seen_original_hashes: set[str] = set()
        seen_prepared_hashes: set[str] = set()
        for record in records:
            if len(rows) >= max_clips:
                break
            file_name = f"{prefix}wav/{record['id']}.wav"
            entry = entries.get(file_name)
            if entry is None:
                rejected_missing += 1
                continue
            if not 1000 <= entry.file_size <= MAX_COMPRESSED_WAV_BYTES:
                rejected_bad += 1
                continue
            try:
                data = zf.read(entry)
                if len(data) > MAX_COMPRESSED_WAV_BYTES:
                    raise ValueError('Oversized WAV')
                src_hash = hashlib.sha256(data).hexdigest()
                if src_hash in seen_original_hashes:
                    rejected_duplicate += 1
                    continue
                source = io.BytesIO(data)
                info = sf.info(source)
                if (info.samplerate != 48000 or info.channels != 1 or info.subtype != 'PCM_16'
                        or not 0.4 <= info.duration <= 20):
                    raise ValueError('Expected 48kHz mono PCM16 WAV between 0.4 and 20s')
                source.seek(0)
                signal, sr = sf.read(source, dtype='float32')
                if signal.ndim != 1 or not np.isfinite(signal).all():
                    raise ValueError('Non-finite or non-mono signal')
                peak = float(np.max(np.abs(signal)))
                rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
                clip_fraction = float(np.mean(np.abs(signal) >= 0.999))
                if peak <= 0.001 or rms < 0.002 or clip_fraction > 0.005:
                    raise ValueError('Silence or clipping')
                prepared = soxr.resample(signal, sr, 22050)
                if not np.isfinite(prepared).all():
                    raise ValueError('Bad resampling')
                out_name = f'{src_hash[:24]}.wav'
                destination = output / 'audio' / out_name
                sf.write(destination, prepared, 22050, subtype='PCM_16')
                prepared_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
                if prepared_hash in seen_prepared_hashes:
                    destination.unlink(missing_ok=True)
                    rejected_duplicate += 1
                    continue
                seen_original_hashes.add(src_hash)
                seen_prepared_hashes.add(prepared_hash)
            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, EOFError):
                rejected_bad += 1
                continue
            rows.append({
                'audio_path': 'audio/' + out_name,
                'original_dataset_row': record['id'],
                'text': record['text'],
                '_group': record['group'],
                'speaker_id': 'adam_synthetic_preset',
                'speaker_type': 'synthetic_voice_preset',
                'speech_origin': 'synthetic',
                'source_audio_sample_rate': 48000,
                'dataset_source': SOURCE_NAME,
                'license': LICENSE,
                'source_url': SOURCE_URL,
                'attribution': 'Phitran21 / Vietnamese-Adam-TTS-50h; source voice rights must be independently reviewed',
                'source_sha256': src_hash,
                'transcript_verified': False,
                'alignment_verified': False,
                'emotion_status': 'unverified',
                'voice_rights_reviewed_by_operator': upstream_voice_rights_reviewed,
                'rights_verified': rights_reviewed and upstream_voice_rights_reviewed,
                'vocoder_training_approved': training_approved,
            })

    # Whole accepted dataset determines group-disjoint split. Train/dev share
    # the *synthetic preset*, so speaker-disjoint validation is impossible.
    groups = sorted({row['_group'] for row in rows}, key=lambda text: _rank(seed, 'group:' + text))
    if len(groups) < 3:
        raise ValueError('Not enough distinct valid transcripts for train/dev split')
    n_dev = max(1, min(len(groups) - 2, round(len(groups) * dev_fraction)))
    dev_groups = set(groups[:n_dev])
    for row in rows:
        row['split'] = 'dev' if row.pop('_group') in dev_groups else 'train'
    train_count = sum(row['split'] == 'train' for row in rows)
    dev_count = sum(row['split'] == 'dev' for row in rows)
    if train_count < 2 or dev_count < 1:
        raise ValueError('Not enough usable samples for 2 train and 1 dev')

    manifest = output / 'vocoder.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    # Keep a separate unapproved transcription review queue for future CTC/MAS.
    # Audio/text pairing is NOT evidence that the words or timings are correct.
    transcript_rows = []
    for row in rows:
        transcript_rows.append({
            'text': row['text'], 'audio_path': row['audio_path'],
            'speaker_id': row['speaker_id'], 'speaker_status': 'synthetic_preset',
            'emotion': 'neutral', 'emotion_status': 'unverified_placeholder',
            'emotion_intensity': 0.5, 'tokenizer': 'grapheme',
            'split': 'review_unassigned',
            'transcript_verified': False, 'alignment_verified': False,
            'dataset_source': SOURCE_NAME, 'source_sha256': row['source_sha256'],
            'source_url': SOURCE_URL, 'license': LICENSE,
            'attribution': row['attribution'],
            'speech_origin': 'synthetic',
        })
    (output / 'transcripts_review.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in transcript_rows),
        encoding='utf-8')
    report = {
        'source_dataset': SOURCE_URL, 'license_label': LICENSE,
        'speech_origin': 'synthetic', 'speaker_count': 1,
        'train_dev_policy': 'text_group_disjoint_single_synthetic_voice',
        'clips_prepared': len(rows), 'train_clips': train_count, 'dev_clips': dev_count,
        'rejected_invalid_transcript': len(rejected_transcripts),
        'rejected_missing_audio': rejected_missing,
        'rejected_bad_audio': rejected_bad,
        'rejected_duplicate_audio': rejected_duplicate,
        'rights_reviewed_by_operator': rights_reviewed,
        'upstream_voice_rights_reviewed_by_operator': upstream_voice_rights_reviewed,
        'training_approved_by_operator': training_approved,
        'limitations': [
            'Synthetic Adam voice, NOT a real human speaker or actor recording',
            'Dataset CC BY 4.0 label does not prove upstream voice rights; review independently',
            'Text/source pronunciations and model quality have not been human-verified',
            'Single synthetic voice: dev split is utterance-disjoint, NOT speaker-disjoint',
            'WAV technical QA does not establish intelligible Vietnamese TTS',
        ],
    }
    (output / 'import_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Prepare synthetic Adam TTS WAVs from local ZIP for VietDub vocoder')
    parser.add_argument('--archive', required=True, help='Local finnal_tts.zip; no network download')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-clips', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--dev-fraction', type=float, default=0.05)
    parser.add_argument('--acknowledge-dataset-rights-reviewed', action='store_true')
    parser.add_argument('--acknowledge-upstream-voice-rights-reviewed', action='store_true')
    parser.add_argument('--approve-vocoder-training', action='store_true')
    args = parser.parse_args(argv)
    report = prepare_adam_tts(
        args.archive, args.output_dir, max_clips=args.max_clips, seed=args.seed,
        dev_fraction=args.dev_fraction, rights_reviewed=args.acknowledge_dataset_rights_reviewed,
        upstream_voice_rights_reviewed=args.acknowledge_upstream_voice_rights_reviewed,
        training_approved=args.approve_vocoder_training)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
