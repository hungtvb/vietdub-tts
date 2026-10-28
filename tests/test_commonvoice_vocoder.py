"""Common Voice importer fixtures are fake synthetic recordings, not actual speech."""
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

from vietdub_native.commonvoice_vocoder import prepare_commonvoice
from vietdub_native.vocoder_corpus import CorpusError, read_vocoder_manifest


def _fake_commonvoice(tmp_path: Path, *, speakers=4, utterances=2) -> Path:
    source = tmp_path / 'commonvoice'
    (source / 'clips').mkdir(parents=True)
    rows = []
    for person in range(speakers):
        for item in range(utterances):
            name = f'p{person}-{item}.mp3'
            wav = tmp_path / f'p{person}-{item}.wav'
            samples = np.arange(18000) / 22050
            sf.write(wav, (0.15*np.sin(2*np.pi*(120+person*80+item*5)*samples)).astype('float32'), 22050)
            command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                       '-i', str(wav), '-codec:a', 'libmp3lame', str(source/'clips'/name)]
            subprocess.run(command, check=True, capture_output=True)
            rows.append({'client_id': f'speaker_{person}', 'path': name, 'sentence': 'xin chào'})
    with (source / 'validated.tsv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=['client_id','path','sentence'], delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    return source


def test_import_without_explicit_approval_remains_blocked(tmp_path):
    root=_fake_commonvoice(tmp_path)
    target=tmp_path/'prepared'
    report=prepare_commonvoice(root, target, max_clips=8)
    assert report['clips_prepared'] == 8
    manifest=target/'vocoder.jsonl'
    with pytest.raises(CorpusError, match='rights_verified'):
        read_vocoder_manifest(manifest, split='train')
    assert all(x['rights_verified'] is False and x['vocoder_training_approved'] is False
               for x in (json.loads(s) for s in manifest.read_text().splitlines()))


def test_approved_manifest_is_speaker_disjoint_and_trainable(tmp_path):
    root=_fake_commonvoice(tmp_path)
    target=tmp_path/'prepared'
    report=prepare_commonvoice(root, target, max_clips=8,
                               rights_reviewed=True, training_approved=True)
    rows=[json.loads(x) for x in (target/'vocoder.jsonl').read_text().splitlines()]
    per_speaker={}
    for row in rows:
        # Local fixture filenames encode speaker only to verify separation; output has no raw client ID.
        speaker=row['source_path'].split('-')[0]
        per_speaker.setdefault(speaker,set()).add(row['split'])
        assert Path(row['audio_path']).is_relative_to('audio')
        assert row['license'] == 'CC0-1.0'
        assert 'client_id' not in row
    assert all(len(x)==1 for x in per_speaker.values())
    assert report['train_speakers'] >= 2 and report['dev_speakers'] >= 1
    assert len(read_vocoder_manifest(target/'vocoder.jsonl', split='train')) >= 2
    assert len(read_vocoder_manifest(target/'vocoder.jsonl', split='dev')) >= 1


def test_source_path_escape_is_rejected_before_decode(tmp_path):
    root=_fake_commonvoice(tmp_path)
    with (root/'validated.tsv').open('a',encoding='utf-8') as f:
        f.write('speaker_extra\t../../secret.mp3\thello\n')
    with pytest.raises(ValueError, match='unsafe|outside|path'):
        prepare_commonvoice(root,tmp_path/'prepared',max_clips=100)
    assert not (tmp_path/'prepared'/'vocoder.jsonl').exists()


def test_rejects_insufficient_distinct_speakers(tmp_path):
    root=_fake_commonvoice(tmp_path,speakers=2)
    with pytest.raises(ValueError, match='speakers'):
        prepare_commonvoice(root,tmp_path/'prepared',max_clips=4)


def test_approved_commonvoice_import_can_start_actual_vocoder_trainer(tmp_path):
    """Integration only uses fake sine MP3 fixtures: never claim this as speech."""
    from vietdub_native.config import Config
    from vietdub_native.hifigan_train import train_hifigan
    root = _fake_commonvoice(tmp_path)
    target = tmp_path/'prepared'
    prepare_commonvoice(root, target, max_clips=8,
                        rights_reviewed=True, training_approved=True)
    result = train_hifigan(target/'vocoder.jsonl', tmp_path/'trained', steps=1,
                           cfg=Config(n_fft=256,n_mels=16), crop_frames=16,
                           generator_channels=16, discriminator_channels=8)
    assert result['steps'] == 1
    assert result['intelligibility_verified'] is False
    assert (tmp_path/'trained'/'dev_reconstructed.wav').exists()


def test_near_silent_mp3_is_excluded_before_vocoder_training(tmp_path):
    root = _fake_commonvoice(tmp_path)
    wav = tmp_path/'silent.wav'
    sf.write(wav, np.zeros(18000, dtype='float32'), 22050)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y',
                    '-i',str(wav),'-codec:a','libmp3lame',str(root/'clips'/'p0-0.mp3')],
                   check=True, capture_output=True)
    report = prepare_commonvoice(root,tmp_path/'prepared', max_clips=8)
    assert report['clips_prepared'] == 7
    assert report['rejected_low_quality'] == 1


def test_tiny_bounded_import_retains_two_distinct_train_speakers(tmp_path):
    root = _fake_commonvoice(tmp_path, speakers=3, utterances=3)
    report = prepare_commonvoice(root, tmp_path/'prepared', max_clips=3, seed=4)
    assert report['train_clips'] == 2 and report['dev_clips'] == 1
    assert report['train_speakers'] == 2
    assert report['dev_speakers'] == 1
