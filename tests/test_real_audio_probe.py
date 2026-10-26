"""All local tests synthetic; the networked Action is the real-audio evidence."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vietdub_native import real_audio_probe as p


class Reply(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()


def _fake_audio():
    sr = 22050
    samples = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype('float32')
    out = io.BytesIO()
    sf.write(out, samples, sr, format='WAV', subtype='PCM_16')
    return out.getvalue()


def test_reject_untrusted_audio_urls():
    for src in ('http://example.org/clip.mp3', 'file:///tmp/test.wav',
                'https://evil.example/clip.mp3', '//evil.example/clip.mp3'):
        with pytest.raises(p.ProbeError):
            p._audio_url({'src': src})
    assert p._audio_url({'src': '/cached-assets/foo'}) == p.SERVER + '/cached-assets/foo'
    assert p._audio_url([{'src': '/cached-assets/foo'}]) == p.SERVER + '/cached-assets/foo'
    with pytest.raises(p.ProbeError, match='Ambiguous'):
        p._audio_url([{'src': '/a'}, {'src': '/b'}])


def test_requires_rights_and_bounded_size(tmp_path):
    with pytest.raises(p.ProbeError, match='license'):
        p.probe(tmp_path / 'out', rights_acknowledged=False)
    for limit in (0, 13):
        with pytest.raises(p.ProbeError, match='Limit'):
            p.probe(tmp_path / 'out', rights_acknowledged=True, limit=limit)


def test_bounded_stream(monkeypatch):
    monkeypatch.setattr(p.request, 'urlopen', lambda req, timeout: Reply(b'012345678'))
    with pytest.raises(p.ProbeError, match='exceeded'):
        p._read('https://datasets-server.huggingface.co/foo', size_limit=4)


def test_empty_or_malformed_viewer_responses(monkeypatch):
    monkeypatch.setattr(p, '_get_json', lambda *_: {'splits': []})
    with pytest.raises(p.ProbeError, match='train split'):
        p._find_split()
    monkeypatch.setattr(p, '_get_json', lambda *_: {'rows': 25})
    with pytest.raises(p.ProbeError, match='did not return'):
        p._load_rows('default','train',5)


def test_end_to_end_probe_mocked_audio(tmp_path, monkeypatch):
    audio = _fake_audio()
    rows = [{'row_idx': i, 'row': {
        'text': f'xin chào {i}', 'audio': [{'src': f'{p.SERVER}/asset-{i}.wav'}]}}
        for i in range(3)]
    monkeypatch.setattr(p, '_find_split', lambda: ('default', 'train'))
    monkeypatch.setattr(p, '_load_rows', lambda *args: rows)
    monkeypatch.setattr(p, '_read', lambda *args, **kwargs: audio)
    # ffmpeg sniffing from arbitrary .bin input is covered here too.
    out = tmp_path / 'real-review'
    report = p.probe(out, rights_acknowledged=True, limit=3, min_ok=3)
    assert report['downloaded_decoded'] == 3
    assert report['training_approved'] is False
    assert report['f0_measured_clips'] == 3
    assert 180 < report['median_f0_hz'] < 260
    parsed = [json.loads(s) for s in (out/'transcripts.jsonl').read_text().splitlines()]
    assert all(row['speaker_id'] == 'unassigned' for row in parsed)
    assert all(row['alignment_verified'] is False for row in parsed)
    assert all(row['transcript_verified'] is False for row in parsed)
    assert all((out/row['audio_path']).is_file() for row in parsed)
    with pytest.raises(p.ProbeError, match='not empty'):
        p.probe(out, rights_acknowledged=True)


def test_unusable_audio_fails_with_report(tmp_path, monkeypatch):
    monkeypatch.setattr(p, '_find_split', lambda: ('default', 'train'))
    monkeypatch.setattr(p, '_load_rows', lambda *args: [{'row_idx': 7, 'row': {'text': 'xin chào'}}])
    with pytest.raises(p.ProbeError, match='need 1'):
        p.probe(tmp_path/'out', rights_acknowledged=True, limit=1, min_ok=1)
    report = json.loads((tmp_path/'out'/'report.json').read_text())
    assert report['downloaded_decoded'] == 0
