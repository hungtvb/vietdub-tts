"""Prepare locally licensed FOSD audio and publisher timestamp TXT for human review.

Published rows: file|transcript|start-end [start-end ...].
One transcript containing multiple speech intervals is AMBIGUOUS and rejected.
Never fabricate actor identity, token timing, emotion labels or rights.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import soundfile as sf
import soxr
from .data_sources import IntakeError, SAMPLE_RATE, _safe_source_path, _wav_samples, inspect_audio

_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
_SECONDS = re.compile(r"^\d+(?:\.\d+)?$")


def parse_seconds(raw: str) -> float:
    """Parse nonnegative decimal seconds or [HH:]MM:SS[.sss]."""
    parts = raw.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(_SECONDS.fullmatch(x) for x in parts):
        raise IntakeError("Unrecognized FOSD timestamp")
    numbers = [float(p) for p in parts]
    if len(numbers) == 1:
        value = numbers[0]
    elif len(numbers) == 2:
        if numbers[1] >= 60:
            raise IntakeError("Timestamp seconds must be < 60")
        value = numbers[0] * 60 + numbers[1]
    else:
        if numbers[1] >= 60 or numbers[2] >= 60:
            raise IntakeError("Timestamp minutes/seconds must be < 60")
        value = numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    if not np.isfinite(value):
        raise IntakeError("Non-finite FOSD timestamp")
    return value


def parse_record(line: str) -> tuple[str, str, tuple[float, float] | None]:
    """Accept one explicitly mapped segment or a whole-recording transcript."""
    parts = line.strip("\ufeff\r\n ").split("|", 2)
    if len(parts) < 2:
        raise IntakeError("FOSD row needs filename|transcript[|interval]")
    filename, spoken = parts[0].strip(), parts[1].strip()
    if not filename or not spoken:
        raise IntakeError("FOSD filename and transcript cannot be empty")
    if len(parts) == 2 or not parts[2].strip():
        return filename, spoken, None
    windows = parts[2].split()
    if len(windows) != 1:
        raise IntakeError("Multiple utterance intervals need manual transcript splitting")
    start_end = windows[0].split("-")
    if len(start_end) != 2:
        raise IntakeError("One FOSD interval must have start-end")
    start, end = map(parse_seconds, start_end)
    if end <= start:
        raise IntakeError("FOSD segment end must be later than start")
    return filename, spoken, (start, end)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_fosd(
    *, transcript_files: list[str | Path], audio_root: str | Path,
    output_dir: str | Path, rights_acknowledged: bool, terms_acknowledged: bool,
    min_seconds: float = 0.4, max_seconds: float = 20.0,
) -> dict:
    """Cut original audio into clips with transcript review CSV, never training approval."""
    if not rights_acknowledged or not terms_acknowledged:
        raise IntakeError("Review FOSD license and source notice before preparing corpus")
    if min_seconds <= 0 or max_seconds <= min_seconds:
        raise IntakeError("Invalid clip duration thresholds")
    root, target = Path(audio_root).resolve(), Path(output_dir).resolve()
    if not root.is_dir():
        raise IntakeError("Audio root does not exist")
    if target == root or target.is_relative_to(root):
        raise IntakeError("Output directory must be separate from original recordings")
    if not transcript_files:
        raise IntakeError("At least one FOSD transcript is required")
    transcripts = [Path(f).resolve() for f in transcript_files]
    if any(not f.is_file() for f in transcripts):
        raise IntakeError("A FOSD transcript file is missing")
    if target.exists() and any(target.iterdir()):
        raise IntakeError("Output folder is not empty; refusing to mix previous runs")
    target.mkdir(parents=True, exist_ok=True)
    clips = target / "clips"
    clips.mkdir()
    results, rejected = [], []
    sha_seen = set()
    hash_cache = {}

    for transcript in transcripts:
        for line_num, raw in enumerate(transcript.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                source_name, text, interval = parse_record(raw)
                file = _safe_source_path(root, source_name)
                if file.suffix.lower() not in _EXTENSIONS:
                    raise IntakeError("Unsupported FOSD audio extension")
                samples, sr = _wav_samples(file)
                if samples.ndim != 2 or not 1 <= samples.shape[1] <= 2 or not 8000 <= sr <= 192000:
                    raise IntakeError("Unsupported audio channels or sample rate")
                duration = samples.shape[0] / sr
                start, end = interval if interval is not None else (0.0, duration)
                if start < 0 or end > duration + .005:
                    raise IntakeError("Transcript time interval exceeds recording duration")
                duration_sec = end-start
                if not min_seconds <= duration_sec <= max_seconds:
                    raise IntakeError("FOSD segment outside accepted duration range")
                chunk = samples[int(round(start*sr)):int(round(min(end, duration)*sr))]
                if not len(chunk):
                    raise IntakeError("Empty FOSD segment")
                mono = np.asarray(chunk.mean(axis=1), dtype=np.float32)
                if sr != SAMPLE_RATE:
                    mono = np.asarray(soxr.resample(mono, sr, SAMPLE_RATE), dtype=np.float32)
                if file not in hash_cache:
                    hash_cache[file] = _sha256_file(file)
                key = hash_cache[file]
                identity = f"{key}|{start:.6f}|{end:.6f}"
                sha = hashlib.sha256(identity.encode("ascii")).hexdigest()
                if sha in sha_seen:
                    raise IntakeError("Repeated FOSD source and time interval")
                candidate = clips / (sha[:32] + ".wav")
                sf.write(candidate, mono, SAMPLE_RATE, subtype="PCM_16")
                try:
                    _, metrics = inspect_audio(candidate, min_seconds=min_seconds, max_seconds=max_seconds)
                except Exception:
                    candidate.unlink(missing_ok=True)
                    raise
                sha_seen.add(sha)
                results.append({
                    "audio_path": candidate.name,
                    "text": text,
                    "speaker_id": "",
                    "original_file": str(Path(source_name)),
                    "start_sec": f"{start:.6f}",
                    "end_sec": f"{end:.6f}",
                    "transcript_status": "needs_human_review",
                    "source_sha256": key,
                    "duration_sec": metrics["duration_sec"],
                })
            except (ValueError, OSError, RuntimeError) as ex:
                rejected.append({"transcript_file": transcript.name, "line": line_num,
                                 "reason": str(ex)[:300]})

    with (target / "metadata_for_review.csv").open("w", encoding="utf-8", newline="") as f:
        keys = ("audio_path", "text", "speaker_id", "original_file", "start_sec",
                "end_sec", "transcript_status", "source_sha256", "duration_sec")
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    with (target / "rejected.jsonl").open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "fpt_fosd_v4", "accepted": len(results), "rejected": len(rejected),
        "duration_hours": round(sum(float(r["duration_sec"]) for r in results) / 3600, 6),
        "original_recordings_are_not_in_repo": True,
        "license_and_voice_rights_verified_by_software": False,
        "rights_and_terms_attested_by_operator": True,
        "dataset_status": "STAGED_FOR_TRANSCRIPT_AND_SPEAKER_REVIEW",
        "ready_for_tts_training": False,
        "notes": "Split multi-interval transcripts by hand; assign real speaker IDs and verify source rights before importing.",
    }
    (target / "prepare_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vietdub-fosd", description="Offline FOSD timestamp preparation for human review"
    )
    parser.add_argument("--transcript", required=True, action="append", dest="transcript_files")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acknowledge-rights", action="store_true")
    parser.add_argument("--acknowledge-terms", action="store_true")
    parser.add_argument("--min-seconds", type=float, default=.4)
    parser.add_argument("--max-seconds", type=float, default=20.)
    args = parser.parse_args(argv)
    data = vars(args)
    data["rights_acknowledged"] = data.pop("acknowledge_rights")
    data["terms_acknowledged"] = data.pop("acknowledge_terms")
    print(json.dumps(prepare_fosd(**data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
