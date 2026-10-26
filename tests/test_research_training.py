"""Only mathematical/structural smoke tests. Synthetic samples are NOT speech."""
from pathlib import Path
import json
import numpy as np
import pytest
import soundfile as sf
from vietdub_native.config import Config
from vietdub_native.research_training import (ResearchError, load_research_samples,
                                               run_research_demo)


def _manifest(tmp_path):
    sr=22050
    audio=[]
    for i in range(2):
        wave=(.08*np.sin(2*np.pi*(160+25*i)*np.arange(int(.65*sr))/sr)).astype('float32')
        sf.write(tmp_path/f'{i}.wav',wave,sr)
        audio.append(dict(text='xin chào',audio_path=f'{i}.wav',speaker_id='unassigned',
                          split='review_unassigned',transcript_verified=False,
                          alignment_verified=False, emotion='neutral',
                          emotion_status='unverified_placeholder',
                          dataset_source='fpt_fosd_v4',dataset_mirror='awghuku/fosd'))
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in audio))
    return manifest


def test_needs_license_review(tmp_path):
    m=_manifest(tmp_path)
    with pytest.raises(ResearchError, match='rights review'):
        run_research_demo(m,tmp_path/'out')


def test_reject_verified_or_unknown_source(tmp_path):
    m=_manifest(tmp_path)
    rows=[json.loads(s) for s in m.read_text().splitlines()]
    rows[0]['transcript_verified']=True
    m.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ResearchError,match='unapproved'):
        load_research_samples(m, Config(n_fft=256,n_mels=16,hidden=16,encoder_layers=1,decoder_layers=1))


def test_weak_supervision_synthesizes_new_wav_but_does_not_certify_quality(tmp_path):
    m=_manifest(tmp_path)
    cfg=Config(n_fft=256,n_mels=16,hidden=16,encoder_layers=1,decoder_layers=1,max_frames=200)
    report=run_research_demo(m,tmp_path/'out',rights_reviewed=True,ctc_steps=1,
                            acoustic_steps=1,vocoder_steps=1,max_clips=2,cfg=cfg)
    wav=tmp_path/'out'/'vietdub-tts-demo.wav'
    x,sr=sf.read(wav)
    assert sr==44100 and x.size>100 and np.isfinite(x).all()
    assert report['clip_count']==2
    assert report['training_approved'] is False
    assert report['intelligibility_verified'] is False
    assert isinstance(report['acoustic_signal_present'], bool)
    assert 'generated_dc_ratio' in report
    assert not (tmp_path/'out'/'transcripts.jsonl').exists()


def test_unintelligible_demo_never_released(tmp_path):
    m=_manifest(tmp_path)
    cfg=Config(n_fft=256,n_mels=16,hidden=16,encoder_layers=1,decoder_layers=1,max_frames=200)
    r=run_research_demo(m,tmp_path/'qa',rights_reviewed=True,ctc_steps=1,acoustic_steps=1,vocoder_steps=1,max_clips=2,cfg=cfg)
    assert r['candidate_for_listening'] is False
    assert r['release_approved'] is False
