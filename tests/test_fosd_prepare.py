"""Regression tests using synthetic audio; NOT real FOSD evidence."""
from __future__ import annotations
import csv
import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from vietdub_native.data_sources import SAMPLE_RATE, import_corpus
from vietdub_native.fosd_prepare import (
    IntakeError, parse_record, parse_seconds, prepare_fosd, main
)


def _setup(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    sf.write(root / "one.wav", .15 * np.sin(2 * np.pi * 220 * t), SAMPLE_RATE)
    return root


def _prepare(tmp_path, txt, *, root=None, output=None):
    return prepare_fosd(
        transcript_files=[txt], audio_root=root or tmp_path / "audio",
        output_dir=output or tmp_path / "staged",
        rights_acknowledged=True, terms_acknowledged=True
    )


def test_timestamp_formats():
    assert parse_seconds("1.35") == 1.35
    assert parse_seconds("01:02.5") == 62.5
    assert parse_seconds("01:02:03.5") == 3723.5
    for x in ("", "1:70", "-3", "00:60:00", "abc"):
        with pytest.raises(IntakeError):
            parse_seconds(x)


def test_fosd_one_interval_or_full_clip_only():
    assert parse_record("one.mp3|Tôi ở đây|0.5-1.25") == (
        "one.mp3", "Tôi ở đây", (.5, 1.25)
    )
    assert parse_record("one.mp3|Xin chào") == ("one.mp3", "Xin chào", None)
    with pytest.raises(IntakeError, match="Multiple utterance"):
        parse_record("one.mp3|Xin chào|0-1 1.5-2")
    with pytest.raises(IntakeError, match="later"):
        parse_record("one.mp3|Xin chào|1-1")


def test_cut_audio_and_keep_speakers_unverified(tmp_path):
    root = _setup(tmp_path)
    txt = tmp_path / "fosd.txt"
    txt.write_text("one.wav|Xin chào|0.4-1.4\none.wav|Tạm biệt|1.6-2.8\n", encoding="utf-8")
    stage = tmp_path / "staged"
    result = _prepare(tmp_path, txt, root=root)
    assert (result["accepted"], result["rejected"]) == (2, 0)
    assert result["ready_for_tts_training"] is False
    with (stage / "metadata_for_review.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["text"] for r in rows] == ["Xin chào", "Tạm biệt"]
    assert rows[0]["speaker_id"] == rows[1]["speaker_id"] == ""
    assert [round(float(r["duration_sec"]), 1) for r in rows] == [1.0, 1.2]
    assert sf.info(stage / "clips" / rows[0]["audio_path"]).samplerate == SAMPLE_RATE
    imported = import_corpus(
        source="fpt_fosd_v4", corpus_format="table",
        metadata=stage / "metadata_for_review.csv", audio_root=stage / "clips",
        output_dir=tmp_path / "imported", speaker_salt="private-random-string-is-long",
        rights_acknowledged=True, terms_acknowledged=True
    )
    assert imported["accepted"] == 2
    assert imported["split_counts"]["review_unassigned"] == 2


def test_reject_ambiguous_intervals_unsafe_paths_missing_audio(tmp_path):
    root = _setup(tmp_path)
    txt = tmp_path / "fosd.txt"
    txt.write_text(
        "one.wav|multiple|0-1 2-3\n"
        "../fosd.txt|unsafe|0-1\n"
        "missing.mp3|missing|0-1\n"
        "one.wav|ok|0-1\n", encoding="utf-8"
    )
    summary = _prepare(tmp_path, txt, root=root)
    assert (summary["accepted"], summary["rejected"]) == (1, 3)
    reasons = (tmp_path / "staged" / "rejected.jsonl").read_text()
    assert "Multiple utterance intervals" in reasons
    assert "outside authorized" in reasons
    assert "audio root" in reasons


def test_silent_audio_or_out_of_bounds_rejected(tmp_path):
    root = _setup(tmp_path)
    sf.write(root / "silent.wav", np.zeros(SAMPLE_RATE * 2), SAMPLE_RATE)
    txt = tmp_path / "fosd.txt"
    txt.write_text(
        "one.wav|too long|0-4\nsilent.wav|silent|0-1\n"
        "one.wav|valid|0-1\n", encoding="utf-8"
    )
    result = _prepare(tmp_path, txt, root=root)
    assert (result["accepted"], result["rejected"]) == (1, 2)
    assert len(list((tmp_path / "staged" / "clips").glob("*.wav"))) == 1


def test_rights_and_nonempty_output_required(tmp_path):
    root = _setup(tmp_path)
    txt = tmp_path / "fosd.txt"
    txt.write_text("one.wav|hello|0-1\n")
    with pytest.raises(IntakeError, match="license"):
        prepare_fosd(
            transcript_files=[txt], audio_root=root,
            output_dir=tmp_path / "staged",
            rights_acknowledged=False, terms_acknowledged=True
        )
    with pytest.raises(IntakeError, match="separate"):
        _prepare(tmp_path, txt, root=root, output=root / "staging")
    _prepare(tmp_path, txt, root=root)
    with pytest.raises(IntakeError, match="not empty"):
        _prepare(tmp_path, txt, root=root)


def test_cli_accepts_transcript(tmp_path, capsys):
    root = _setup(tmp_path)
    txt = tmp_path / "fosd.txt"
    txt.write_text("one.wav|Chào bạn|0-1\n", encoding="utf-8")
    main(["--transcript", str(txt), "--audio-root", str(root),
          "--output-dir", str(tmp_path / "result"),
          "--acknowledge-rights", "--acknowledge-terms"])
    assert json.loads(capsys.readouterr().out)["accepted"] == 1


def test_mp3_decoding_when_ffmpeg_available(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg not installed")
    root = _setup(tmp_path)
    import subprocess
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-i",
        str(root / "one.wav"), str(root / "one.mp3")
    ], check=True)
    txt = tmp_path / "fosd.txt"
    txt.write_text("one.mp3|Được rồi|0.5-1.8\n", encoding="utf-8")
    assert _prepare(tmp_path, txt, root=root)["accepted"] == 1
