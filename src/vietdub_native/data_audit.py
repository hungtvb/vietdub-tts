"""Auditable real-speech readiness gate. NEVER infers verification or consent.

The report evaluates asserted metadata and local file integrity, not the actual
legal legitimacy, waveform/word alignment or acting quality of those assertions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import soundfile as sf

from .config import EMOTIONS
from .symbols import TOKENIZERS, encode, unknown_characters

STAGES = ("aligner", "acoustic", "expressive")
SPLITS = ("train", "dev", "test", "review_unassigned")


class AuditError(ValueError):
    """Invalid audit invocation or data manifest."""


def _is_true(value: object) -> bool:
    return value is True


def _valid_intensities(value: object) -> bool:
    return type(value) in (float, int) and 0 <= value <= 1


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise AuditError(f"Missing manifest: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as e:
            raise AuditError(f"{path}:{line_number}: invalid JSON: {e}") from e
        if not isinstance(value, dict):
            raise AuditError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    if not rows:
        raise AuditError(f"Manifest is empty: {path}")
    return rows


def _fingerprint(row: dict, audio: Path) -> str:
    """Deduplicate against source SHA when available; otherwise prepared bytes."""
    source_sha = row.get("source_sha256")
    if isinstance(source_sha, str) and len(source_sha) == 64 and all(c in "0123456789abcdef" for c in source_sha.lower()):
        return "source:" + source_sha.lower()
    digest = hashlib.sha256()
    with audio.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "prepared:" + digest.hexdigest()


def _audio_details(audio: Path) -> tuple[int, int]:
    details = sf.info(str(audio))
    if not details.frames or not details.samplerate:
        raise AuditError("Unusable audio file")
    return int(details.frames), int(details.samplerate)


def audit_manifests(manifests: Iterable[str | Path], *, stage: str = "aligner",
                    strict: bool = False) -> dict:
    """Inspect datasets locally without editing them or approving any recording.

    strict=True additionally raises when any training-split row is not ready.
    The returned JSON report never includes personal/raw speaker identifiers.
    """
    if stage not in STAGES:
        raise AuditError(f"Unknown audit stage: {stage}")
    paths = [Path(p).resolve() for p in manifests]
    if not paths:
        raise AuditError("Provide at least one manifest")
    if len(set(paths)) != len(paths):
        raise AuditError("Repeated manifest path")
    records: list[dict] = []
    source_seen = defaultdict(list)
    speakers_by_split = defaultdict(set)
    split_speakers = defaultdict(set)
    speaker_emotions = defaultdict(set)
    rejected_reasons = Counter()
    source_counts = Counter()
    split_counts = Counter()
    for path in paths:
        for line_index, row in enumerate(_read_jsonl(path), 1):
            why = []
            def deny(reason: str) -> None:
                if reason not in why:
                    why.append(reason)
            source = row.get("dataset_source", "unknown")
            source = source if isinstance(source, str) and source else "unknown"
            source_counts[source] += 1
            split = row.get("split", "review_unassigned")
            split = split if isinstance(split, str) and split in SPLITS else "review_unassigned"
            split_counts[split] += 1
            speaker = row.get("speaker_id", "unassigned")
            speaker = speaker if isinstance(speaker, str) and speaker.strip() else "unassigned"
            tokenizer = row.get("tokenizer", "grapheme")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                deny("missing_transcript")
            elif tokenizer not in TOKENIZERS:
                deny("unknown_tokenizer")
            else:
                try:
                    if unknown_characters(text, tokenizer):
                        deny("unsupported_transcript_tokens")
                except ValueError:
                    deny("invalid_transcript")
            if not _is_true(row.get("transcript_verified")):
                deny("transcript_not_human_verified")
            if not isinstance(row.get("license"), str) or not row["license"].strip():
                deny("missing_license_provenance")
            if not isinstance(row.get("source_url"), str) or not row["source_url"].strip():
                deny("missing_source_url")
            if not isinstance(row.get("attribution"), str) or not row["attribution"].strip():
                deny("missing_attribution")
            if source == "google_fleurs_vi" and split != "test":
                deny("fleurs_evaluation_only")
            if split == "review_unassigned":
                deny("no_training_split")
            if split == "train" and speaker == "unassigned":
                deny("unknown_training_speaker")
            if speaker not in ("unassigned", "") and split != "review_unassigned":
                speakers_by_split[speaker].add(split)
                split_speakers[split].add(speaker)
            file_path = row.get("audio_path")
            audio = None
            if not isinstance(file_path, str) or not file_path.strip() or "://" in file_path:
                deny("invalid_audio_path")
            else:
                audio = (path.parent / file_path).resolve()
                if not audio.is_file():
                    deny("audio_missing")
                else:
                    try:
                        frames, sr = _audio_details(audio)
                        if sr != 22050:
                            deny("audio_not_22050hz")
                        if frames / sr < 0.4 or frames / sr > 20.0:
                            deny("audio_duration_out_of_range")
                        source_seen[_fingerprint(row, audio)].append(len(records))
                    except (OSError, ValueError, RuntimeError):
                        deny("audio_unreadable")
            emotion = row.get("emotion")
            if stage in ("acoustic", "expressive"):
                if row.get("alignment_verified") is not True:
                    deny("alignment_not_human_verified")
                if not str(row.get("alignment_method", "")).strip():
                    deny("alignment_method_missing")
                durations = row.get("token_durations")
                if tokenizer in TOKENIZERS and isinstance(text, str) and text.strip():
                    try:
                        length = len(encode(text, tokenizer))
                        if not isinstance(durations, list) or len(durations) != length or any(type(d) is not int or d < 0 for d in durations) or sum(durations) <= 0:
                            deny("token_durations_invalid")
                        elif audio is not None and audio.is_file():
                            frames, sr = _audio_details(audio)
                            # Torchaudio MelSpectrogram center=True: 1+floor(wave samples / hop).
                            expected_frames = 1 + frames // 256 if sr == 22050 else -1
                            if sum(durations) != expected_frames:
                                deny("aligned_frame_count_mismatch")
                    except (ValueError, OSError, RuntimeError):
                        deny("token_durations_invalid")
                if speaker == "unassigned" or row.get("speaker_status") == "unassigned":
                    deny("speaker_identity_unverified")
                if emotion not in EMOTIONS:
                    deny("invalid_emotion_label")
                if not _valid_intensities(row.get("emotion_intensity", .5)):
                    deny("invalid_emotion_intensity")
                if row.get("emotion_status") != "human_verified":
                    deny("emotion_not_human_verified")
                if source == "actor_recordings" and not _is_true(row.get("voice_training_consent_verified")):
                    deny("actor_voice_training_consent_not_verified")
                if row.get("emotion_status") == "human_verified" and emotion in EMOTIONS and speaker != "unassigned":
                    speaker_emotions[speaker].add(emotion)
            records.append({"manifest": path.name, "line": line_index, "split": split,
                            "source": source, "speaker": speaker, "reasons": why})
    # Split leakage is global across ALL manifests in this audit call.
    for i in source_seen.values():
        if len(i) > 1:
            for index in i:
                if "duplicate_audio" not in records[index]["reasons"]:
                    records[index]["reasons"].append("duplicate_audio")
    leaky_speakers = {s for s, splits in speakers_by_split.items() if len(splits) > 1}
    for row in records:
        if row["speaker"] in leaky_speakers:
            row["reasons"].append("speaker_split_leakage")
        if stage == "expressive" and row["split"] == "train" and row["speaker"] != "unassigned":
            if len(speaker_emotions[row["speaker"]]) < 2:
                row["reasons"].append("single_emotion_speaker")
        for reason in row["reasons"]:
            rejected_reasons[reason] += 1
    by_split = {name: {"total": sum(r["split"] == name for r in records),
                       "ready": sum(r["split"] == name and not r["reasons"] for r in records)}
                for name in SPLITS}
    # Aggregate excludes pseudonymous speaker values from report; only counts remain.
    report = {
        "stage": stage, "total_rows": len(records),
        "ready_rows": sum(not r["reasons"] for r in records),
        "blocked_rows": sum(bool(r["reasons"]) for r in records),
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "by_split": by_split,
        "distinct_speakers": len(speakers_by_split),
        "leaking_speakers_count": len(leaky_speakers),
        "failure_reasons": dict(sorted(rejected_reasons.items())),
        "findings": [{"manifest": r["manifest"], "line": r["line"],
                      "split": r["split"], "reasons": r["reasons"]}
                     for r in records if r["reasons"]],
        "limitations": [
            "Software cannot verify transcript truth, actor consent, licensing, or emotional authenticity",
            "Speaker split leakage can only be detected for IDs linked across manifests",
            "An alignment frame-count match does not validate phoneme boundaries",
            "Training-readiness metadata is an operator assertion; use manual review",
        ],
    }
    if strict and any(r["split"] == "train" and r["reasons"] for r in records):
        raise AuditError(f"Train split not ready: {report['by_split']['train']}, failures={report['failure_reasons']}")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="vietdub-audit", description="Review real-speech training readiness")
    parser.add_argument("manifests", nargs="+", help="JSONL manifests to audit together")
    parser.add_argument("--stage", choices=STAGES, default="aligner")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on blocked train clips")
    args = parser.parse_args(argv)
    try:
        report = audit_manifests(args.manifests, stage=args.stage)
    except AuditError as exc:
        parser.error(str(exc))
    body = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(body, encoding="utf-8")
    print(body, end="")
    if args.strict and report["by_split"]["train"]["ready"] != report["by_split"]["train"]["total"]:
        parser.exit(2, "Not training-ready: see audit report above\n")


if __name__ == "__main__":
    main()
