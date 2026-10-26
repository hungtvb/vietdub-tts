"""End-to-end and safety tests for local, license-attested corpus import."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vietdub_native.data_sources import (SOURCES, SAMPLE_RATE, IntakeError,
                                         import_corpus, inspect_audio, main)
from vietdub_native.aligner import TranscriptDataset
from vietdub_native.config import Config
from vietdub_native.dataset import AlignedSpeechDataset, ManifestError


def _write_voice(path: Path, *, hz=180, amp=.15, duration=.65):
    n = int(SAMPLE_RATE * duration)
    x = (amp * np.sin(2 * np.pi * hz * np.arange(n) / SAMPLE_RATE)).astype('float32')
    sf.write(path, x, SAMPLE_RATE, subtype='PCM_16')


def _fixtures(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    _write_voice(root / 'one.wav')
    _write_voice(root / 'two.wav', hz=210)
    _write_voice(root / 'three.wav', hz=240)
    metadata = tmp_path / 'validated.tsv'
    with metadata.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'sentence', 'client_id'], delimiter='\t')
        w.writeheader()
        w.writerows([
            {'path':'one.wav', 'sentence':'Xin chào', 'client_id':'a1'},
            {'path':'two.wav', 'sentence':'Tôi ở đây!', 'client_id':'a1'},
            {'path':'three.wav', 'sentence':'Đừng đi!', 'client_id':'a2'},
        ])
    return root, metadata


def _kwargs(tmp_path):
    root, metadata = _fixtures(tmp_path)
    return dict(source='mozilla_common_voice_vi_27', corpus_format='common_voice',
                metadata=metadata, audio_root=root, output_dir=tmp_path/'prepared',
                speaker_salt='this-is-a-private-random-salt',
                rights_acknowledged=True, terms_acknowledged=True)


def test_dataset_catalogue_has_only_checked_sources():
    assert set(SOURCES) == {'mozilla_common_voice_vi_27', 'fpt_fosd_v4',
                            'google_fleurs_vi', 'actor_recordings'}
    assert SOURCES['google_fleurs_vi']['license'] == 'CC-BY-4.0'
    assert SOURCES['mozilla_common_voice_vi_27']['license'] == 'CC0-1.0'


def test_rights_acknowledgement_required(tmp_path):
    args = _kwargs(tmp_path)
    args['rights_acknowledged'] = False
    with pytest.raises(IntakeError, match='Confirm rights'):
        import_corpus(**args)
    args['rights_acknowledged'] = True
    args['terms_acknowledged'] = False
    with pytest.raises(IntakeError, match='Confirm rights'):
        import_corpus(**args)


def test_import_common_voice_pseudonymized_and_ctc_ready_not_acoustic(tmp_path):
    kwargs = _kwargs(tmp_path)
    report = import_corpus(**kwargs)
    assert report['accepted'] == 3 and report['rejected'] == 0
    assert report['duration_hours'] > 0
    assert report['license_and_consents_verified_by_software'] is False
    folder = kwargs['output_dir']
    rows = [json.loads(x) for x in (folder/'transcripts.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    assert rows[0]['speaker_id'] == rows[1]['speaker_id']
    assert rows[0]['speaker_id'] != rows[2]['speaker_id']
    assert rows[0]['speaker_id'] != 'a1'
    assert rows[0]['split'] == rows[1]['split']
    assert {r['alignment_verified'] for r in rows} == {False}
    assert {r['emotion_status'] for r in rows} == {'unverified_placeholder'}
    assert {r['emotion'] for r in rows} == {'neutral'}  # not verified neutral speech
    assert all((folder/r['audio_path']).exists() for r in rows)
    assert all(sf.info(folder/r['audio_path']).samplerate == SAMPLE_RATE for r in rows)
    assert len(TranscriptDataset(folder/'transcripts.jsonl', Config())) == 3
    with pytest.raises(ManifestError):
        AlignedSpeechDataset(folder/'transcripts.jsonl', Config(), [rows[0]['speaker_id']])


def test_reject_path_traversal_and_duplicate_audio(tmp_path):
    args = _kwargs(tmp_path)
    m = args['metadata']
    with m.open('a', encoding='utf-8') as f:
        f.write('../validated.tsv\tnguy hiểm\tbad\n')
        f.write('one.wav\tXin chào\ta1\n')
    report = import_corpus(**args)
    assert report['accepted'] == 3 and report['rejected'] == 2
    rejected = [json.loads(x) for x in (args['output_dir']/'rejected.jsonl').read_text().splitlines()]
    assert any('outside authorized' in r['reason'] for r in rejected)
    assert any('Duplicate' in r['reason'] for r in rejected)


def test_unsupported_text_and_silent_audio_reported(tmp_path):
    args = _kwargs(tmp_path)
    sf.write(args['audio_root'] / 'silence.wav', np.zeros(int(.6*SAMPLE_RATE)), SAMPLE_RATE)
    with args['metadata'].open('a', encoding='utf-8') as f:
        f.write('one.wav\t中文\ta3\n')
        f.write('silence.wav\tXin chào\ta3\n')
    report = import_corpus(**args)
    assert report['accepted'] == 3 and report['rejected'] == 2


def test_missing_speaker_is_unassigned_and_reserved_for_review(tmp_path):
    args = _kwargs(tmp_path)
    text = args['metadata'].read_text()
    args['metadata'].write_text(text.replace('a1', '', 1), encoding='utf-8')
    report = import_corpus(**args)
    rows = [json.loads(x) for x in (args['output_dir']/'transcripts.jsonl').read_text().splitlines()]
    assert report['accepted'] == 3
    assert rows[0]['speaker_id'] == 'unassigned'
    assert rows[0]['split'] == 'review_unassigned'


def test_audio_duration_and_silence_checks(tmp_path):
    f = tmp_path/'short.wav'
    _write_voice(f, duration=.1)
    with pytest.raises(IntakeError, match='Duration'):
        inspect_audio(f)
    _write_voice(f, amp=0)
    with pytest.raises(IntakeError, match='Silent'):
        inspect_audio(f)


def test_metadata_validation_and_output_safety(tmp_path):
    args = _kwargs(tmp_path)
    args['output_dir'] = args['audio_root']/'prepared'
    with pytest.raises(IntakeError, match='separate'):
        import_corpus(**args)
    args['output_dir'] = tmp_path/'prepared'
    args['speaker_salt'] = 'short'
    with pytest.raises(IntakeError, match='random'):
        import_corpus(**args)


def test_generic_table_fpt_with_explicit_speaker(tmp_path):
    args = _kwargs(tmp_path)
    args['source'] = 'fpt_fosd_v4'
    args['corpus_format'] = 'table'
    args['metadata'] = tmp_path/'data.csv'
    args['metadata'].write_text('audio_path,text,speaker_id\none.wav,Xin chào,voice_A\n', encoding='utf-8')
    report = import_corpus(**args)
    assert report['accepted'] == 1 and report['license'].startswith('CC-BY-4.0')


def test_fleurs_holdout_manual_split(tmp_path):
    args = _kwargs(tmp_path)
    args['source'] = 'google_fleurs_vi'
    args['corpus_format'] = 'table'
    args['metadata'] = tmp_path/'data.csv'
    args['metadata'].write_text('audio_path,text,speaker_id\none.wav,Xin chào,voice_A\n', encoding='utf-8')
    args['split'] = 'test'
    report = import_corpus(**args)
    assert report['split_counts']['test'] == 1


def test_cli_sources_and_import(tmp_path, capsys):
    main(['sources'])
    assert 'google_fleurs_vi' in capsys.readouterr().out
    args = _kwargs(tmp_path)
    main(['import', '--source', args['source'], '--format', args['corpus_format'],
          '--metadata', str(args['metadata']), '--audio-root', str(args['audio_root']),
          '--output-dir', str(args['output_dir']), '--speaker-salt', args['speaker_salt'],
          '--acknowledge-rights', '--acknowledge-terms'])
    assert json.loads(capsys.readouterr().out)['accepted'] == 3


def test_fleurs_defaults_to_evaluation(tmp_path):
    args = _kwargs(tmp_path)
    args['source'] = 'google_fleurs_vi'
    args['corpus_format'] = 'table'
    args['metadata'] = tmp_path/'data.csv'
    args['metadata'].write_text('audio_path,text,speaker_id\none.wav,Xin chào,voice_A\n', encoding='utf-8')
    assert import_corpus(**args)['split_counts']['test'] == 1


def test_cli_salt_from_private_environment_variable(tmp_path, monkeypatch, capsys):
    args = _kwargs(tmp_path)
    monkeypatch.setenv('VIETDUB_DATA_SALT', args['speaker_salt'])
    main(['import', '--source', args['source'], '--format', args['corpus_format'],
          '--metadata', str(args['metadata']), '--audio-root', str(args['audio_root']),
          '--output-dir', str(args['output_dir']), '--speaker-salt-env', 'VIETDUB_DATA_SALT',
          '--acknowledge-rights', '--acknowledge-terms'])
    assert json.loads(capsys.readouterr().out)['accepted'] == 3
