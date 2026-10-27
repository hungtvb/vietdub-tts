"""Offline synthetic fixtures. These are NOT examples of intelligible Vietnamese."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vietdub_native.adam_tts_vocoder import prepare_adam_tts
from vietdub_native.vocoder_corpus import CorpusError, read_vocoder_manifest


def _archive(tmp_path: Path, *, texts=None, missing=None, wrong_rate=None, include_attack=False):
    texts = texts or ["Xin chào Việt Nam", "Hôm nay thật đẹp", "Tôi là Adam", "Xin chào Việt Nam", "Chúng ta đi thôi", "Bầu trời xanh", "Thời tiết đẹp", "Tạm biệt"]
    src = tmp_path / 'finnal_tts.zip'
    with zipfile.ZipFile(src, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('finnal_tts/data.jsonl', ''.join(json.dumps({'id': i + 1, 'text': t}, ensure_ascii=False) + '\n' for i, t in enumerate(texts)))
        for i, _ in enumerate(texts, 1):
            if i == missing:
                continue
            rate = 22050 if i == wrong_rate else 48000
            t = np.arange(int(0.8 * rate)) / rate
            wave = (0.12 * np.sin(2 * np.pi * (185 + i * 37) * t)).astype(np.float32)
            b = io.BytesIO()
            sf.write(b, wave, rate, format='WAV', subtype='PCM_16')
            zf.writestr(f'finnal_tts/wav/{i}.wav', b.getvalue())
        if include_attack:
            zf.writestr('../../outside.txt', 'ATTACK')
    return src


def test_review_only_by_default_and_synthetic_provenance(tmp_path):
    source = _archive(tmp_path)
    report = prepare_adam_tts(source, tmp_path / 'prepared', max_clips=6)
    assert report['clips_prepared'] == 6
    assert report['speech_origin'] == 'synthetic'
    assert report['speaker_count'] == 1
    assert report['train_dev_policy'] == 'text_group_disjoint_single_synthetic_voice'
    rows = [json.loads(s) for s in (tmp_path / 'prepared/vocoder.jsonl').read_text().splitlines()]
    assert {r['speaker_id'] for r in rows} == {'adam_synthetic_preset'}
    assert all(r['rights_verified'] is False and r['vocoder_training_approved'] is False for r in rows)
    assert all(r['dataset_source'] == 'phitran21_vietnamese_adam_tts_50h' for r in rows)
    assert all(r['audio_path'].startswith('audio/') and r['license'] == 'CC-BY-4.0' for r in rows)
    with pytest.raises(CorpusError, match='rights_verified'):
        read_vocoder_manifest(tmp_path / 'prepared/vocoder.jsonl', split='train')


def test_approval_requires_upstream_voice_rights_review(tmp_path):
    source = _archive(tmp_path)
    with pytest.raises(ValueError, match='upstream'):
        prepare_adam_tts(source, tmp_path / 'out', max_clips=5,
                         rights_reviewed=True, training_approved=True)
    with pytest.raises(ValueError, match='rights'):
        prepare_adam_tts(source, tmp_path / 'out', max_clips=5,
                         upstream_voice_rights_reviewed=True, training_approved=True)


def test_approved_import_is_vocoder_ready_with_disjoint_text_groups(tmp_path):
    source = _archive(tmp_path)
    root = tmp_path / 'approved'
    report = prepare_adam_tts(source, root, max_clips=8, seed=11,
                              rights_reviewed=True, upstream_voice_rights_reviewed=True,
                              training_approved=True)
    rows = [json.loads(s) for s in (root / 'vocoder.jsonl').read_text().splitlines()]
    train_texts = {r['text'] for r in rows if r['split'] == 'train'}
    dev_texts = {r['text'] for r in rows if r['split'] == 'dev'}
    assert not train_texts.intersection(dev_texts)
    assert report['clips_prepared'] == len(rows) == 8
    assert all(r['source_audio_sample_rate'] == 48000 for r in rows)
    assert all(sf.info(root / r['audio_path']).samplerate == 22050 for r in rows)
    assert len(read_vocoder_manifest(root / 'vocoder.jsonl', split='train')) >= 2
    assert len(read_vocoder_manifest(root / 'vocoder.jsonl', split='dev')) >= 1
    assert all(r['voice_rights_reviewed_by_operator'] is True for r in rows)


def test_bad_clip_rejected_and_never_publishes_unvalidated_audio(tmp_path):
    source = _archive(tmp_path, wrong_rate=1, missing=2)
    root = tmp_path / 'prepared'
    report = prepare_adam_tts(source, root, max_clips=7)
    assert report['rejected_bad_audio'] >= 1
    assert report['rejected_missing_audio'] >= 1
    assert report['clips_prepared'] == 6
    assert not (root / 'audio/1.wav').exists()


def test_rejects_ambiguous_metadata_and_traversal_entries(tmp_path):
    source = _archive(tmp_path, include_attack=True)
    with pytest.raises(ValueError, match='unsafe ZIP member'):
        prepare_adam_tts(source, tmp_path / 'bad', max_clips=5)
    assert not (tmp_path / 'outside.txt').exists()


def test_utterance_split_is_deterministic(tmp_path):
    source = _archive(tmp_path)
    prepare_adam_tts(source, tmp_path / 'first', max_clips=5, seed=21)
    prepare_adam_tts(source, tmp_path / 'second', max_clips=5, seed=21)
    first = [(r['original_dataset_row'], r['split']) for r in map(json.loads,(tmp_path/'first/vocoder.jsonl').read_text().splitlines())]
    second = [(r['original_dataset_row'], r['split']) for r in map(json.loads,(tmp_path/'second/vocoder.jsonl').read_text().splitlines())]
    assert first == second


def test_tiny_limit_has_train_and_dev_without_leaking_repeated_transcript(tmp_path):
    source = _archive(tmp_path)
    report = prepare_adam_tts(source, tmp_path / 'tiny', max_clips=3)
    assert report['train_clips'] >= 2 and report['dev_clips'] >= 1
    rows = [json.loads(s) for s in (tmp_path/'tiny/vocoder.jsonl').read_text().splitlines()]
    assert {r['text'] for r in rows if r['split'] == 'train'}.isdisjoint({r['text'] for r in rows if r['split'] == 'dev'})


def test_approved_adam_import_feeds_native_vocoder_without_alignment(tmp_path):
    """Minimal training integration; sine-wave fixtures do not validate speech quality."""
    from vietdub_native.config import Config
    from vietdub_native.hifigan_train import train_hifigan
    archive = _archive(tmp_path)
    prepared = tmp_path / 'approved'
    prepare_adam_tts(archive, prepared, max_clips=8, rights_reviewed=True,
                     upstream_voice_rights_reviewed=True, training_approved=True)
    report = train_hifigan(prepared / 'vocoder.jsonl', tmp_path / 'run', steps=1,
                           cfg=Config(n_fft=256, n_mels=16), crop_frames=16,
                           generator_channels=16, discriminator_channels=8,
                           eval_seconds=0.5)
    assert report['intelligibility_verified'] is False
    assert (tmp_path / 'run/dev_reconstructed.wav').is_file()


def test_audio_text_pairs_exported_for_future_alignment_but_never_auto_verified(tmp_path):
    source = _archive(tmp_path)
    prepared = tmp_path / 'pair_review'
    prepare_adam_tts(source, prepared, max_clips=6)
    review = prepared / 'transcripts_review.jsonl'
    assert review.is_file()
    rows = [json.loads(line) for line in review.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 6
    assert all(r['transcript_verified'] is False and r['alignment_verified'] is False for r in rows)
    assert all(r['split'] == 'review_unassigned' and 'token_durations' not in r for r in rows)
    assert all(r['emotion_status'] == 'unverified_placeholder' for r in rows)
