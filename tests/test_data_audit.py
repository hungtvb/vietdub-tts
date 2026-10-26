"""Regression tests for training dataset governance, not acting/ASR accuracy."""
import json
import numpy as np
import pytest
import soundfile as sf
from vietdub_native.data_audit import AuditError, audit_manifests, main
from vietdub_native.symbols import encode


def _create(tmp_path, *, audio_name="one.wav", speaker="s1", split="train",
            emotion="neutral", source="fpt_fosd_v4", approved=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sr = 22050
    t = np.arange(int(.65 * sr), dtype=np.float32) / sr
    if not (tmp_path / audio_name).exists():
        sf.write(tmp_path / audio_name, (.15*np.sin(2*np.pi*(210 if audio_name == "one.wav" else 290)*t)), sr)
    row = dict(text="Xin chào", audio_path=audio_name, speaker_id=speaker, split=split,
               dataset_source=source, license="CC-BY-4.0", source_url="https://data.mendeley.com/",
               attribution="FPT 2018", emotion=emotion, emotion_intensity=.4,
               transcript_verified=approved, tokenizer="grapheme",
               alignment_verified=False, alignment_status="not_aligned")
    return row


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
    return path


def test_aligner_stage_requires_manual_transcript_review(tmp_path):
    row = _create(tmp_path, approved=False)
    manifest = _write(tmp_path/'corpus.jsonl', [row])
    report = audit_manifests([manifest])
    assert report['total_rows'] == 1 and report['ready_rows'] == 0
    assert report['failure_reasons']['transcript_not_human_verified'] == 1
    row['transcript_verified'] = True
    _write(manifest, [row])
    ready = audit_manifests([manifest], strict=True)
    assert ready['ready_rows'] == 1


def test_acoustic_needs_alignment_and_emotion_review(tmp_path):
    row = _create(tmp_path)
    manifest = _write(tmp_path/'corpus.jsonl', [row])
    r = audit_manifests([manifest], stage="acoustic")
    assert 'alignment_not_human_verified' in r['failure_reasons']
    assert 'emotion_not_human_verified' in r['failure_reasons']
    assert 'token_durations_invalid' in r['failure_reasons']
    sr = 22050
    frames = sf.info(tmp_path / 'one.wav').frames
    total = frames // 256 + 1
    count = len(encode(row['text']))
    row.update(alignment_verified=True, alignment_method="human_reviewed",
               emotion_status="human_verified",
               token_durations=[total//count+int(i < total%count) for i in range(count)])
    _write(manifest, [row])
    r = audit_manifests([manifest], stage="acoustic")
    assert r['ready_rows'] == 1
    row['token_durations'][0] += 1
    _write(manifest, [row])
    assert 'aligned_frame_count_mismatch' in audit_manifests([manifest], stage="acoustic")['failure_reasons']


def test_duplicate_audio_leakage_within_and_across_manifests(tmp_path):
    row = _create(tmp_path)
    second = dict(row, speaker_id='s2', split='dev')
    first_file = _write(tmp_path/'a.jsonl', [row])
    other_dir = tmp_path/'other'
    other_dir.mkdir()
    (other_dir/'one.wav').write_bytes((tmp_path/'one.wav').read_bytes())
    second_file = _write(other_dir/'b.jsonl', [second])
    report = audit_manifests([first_file, second_file])
    assert report['blocked_rows'] == 2
    assert report['failure_reasons']['duplicate_audio'] == 2


def test_speaker_cross_split_leakage(tmp_path):
    row = _create(tmp_path)
    _create(tmp_path, audio_name="two.wav")
    other = dict(row, audio_path='two.wav', split='test')
    manifest = _write(tmp_path/'m.jsonl', [row, other])
    report = audit_manifests([manifest])
    assert report['leaking_speakers_count'] == 1
    assert report['failure_reasons']['speaker_split_leakage'] == 2


def test_fleurs_is_test_only(tmp_path):
    row = _create(tmp_path, split='train', source='google_fleurs_vi')
    manifest = _write(tmp_path/'m.jsonl', [row])
    report = audit_manifests([manifest])
    assert report['failure_reasons']['fleurs_evaluation_only'] == 1


def test_expressive_requires_multistyle_same_actor(tmp_path):
    row = _create(tmp_path)
    count = len(encode(row['text']))
    frames = sf.info(tmp_path / 'one.wav').frames // 256 + 1
    row.update(alignment_verified=True, alignment_method='manual_review',
               emotion_status='human_verified', token_durations=[frames//count+int(i<frames%count) for i in range(count)])
    first = _write(tmp_path/'manifest.jsonl', [row])
    one = audit_manifests([first], stage="expressive")
    assert 'single_emotion_speaker' in one['failure_reasons']
    _create(tmp_path, audio_name="two.wav")
    second = dict(row, audio_path='two.wav', emotion='sad')
    _write(first, [row, second])
    assert audit_manifests([first], stage="expressive")['ready_rows'] == 2


def test_custom_actor_consent_is_mandatory(tmp_path):
    row = _create(tmp_path, source='actor_recordings')
    count = len(encode(row['text']))
    frames = sf.info(tmp_path / 'one.wav').frames // 256 + 1
    row.update(alignment_verified=True, alignment_method='manual_review', emotion_status='human_verified',
               token_durations=[frames//count+int(i<frames%count) for i in range(count)])
    manifest = _write(tmp_path/'m.jsonl', [row])
    report = audit_manifests([manifest], stage='acoustic')
    assert 'actor_voice_training_consent_not_verified' in report['failure_reasons']


def test_missing_audio_blocks_training(tmp_path):
    row = _create(tmp_path)
    (tmp_path/'one.wav').unlink()
    r = audit_manifests([_write(tmp_path/'m.jsonl', [row])])
    assert r['failure_reasons']['audio_missing'] == 1


def test_cli_json_and_strict_exit(tmp_path, capsys):
    row = _create(tmp_path, approved=False)
    manifest = _write(tmp_path/'m.jsonl', [row])
    output = tmp_path/'qa'/'report.json'
    main([str(manifest), '--output', str(output)])
    assert json.loads(output.read_text())['blocked_rows'] == 1
    with pytest.raises(SystemExit) as exc:
        main([str(manifest), '--strict'])
    assert exc.value.code == 2


def test_duplicate_paths_are_rejected(tmp_path):
    manifest = _write(tmp_path/'m.jsonl', [_create(tmp_path)])
    with pytest.raises(AuditError, match='Repeated'):
        audit_manifests([manifest, manifest])


def test_real_corpus_train_aligner_refuses_unverified_transcript(tmp_path):
    from vietdub_native.aligner import AlignmentError, train_aligner
    from vietdub_native.config import Config
    row = _create(tmp_path, approved=False)
    manifest = _write(tmp_path/'raw.jsonl', [row])
    with pytest.raises(AlignmentError, match='not approved'):
        train_aligner(manifest, tmp_path/'ctc.pt',
                      cfg=Config(n_fft=256, n_mels=16, hidden=32, encoder_layers=1,
                                 decoder_layers=1, max_frames=500), epochs=1)
    assert not (tmp_path/'ctc.pt').exists()


def test_real_corpus_train_acoustic_refuses_unreviewed_emotions(tmp_path):
    from vietdub_native.train import train_acoustic
    from vietdub_native.config import Config
    row = _create(tmp_path)
    manifest = _write(tmp_path/'raw.jsonl', [row])
    with pytest.raises((ValueError, RuntimeError), match='(token_durations|aligned)'):
        train_acoustic(manifest, tmp_path/'model', cfg=Config(hidden=32, n_fft=256,
                       n_mels=16, encoder_layers=1, decoder_layers=1), epochs=1)
    assert not (tmp_path/'model'/'acoustic.pt').exists()
