"""Rights-aware, offline Vietnamese speech corpus intake.

This converts user-supplied LOCAL corpus copies into unaligned transcripts for
VietDub's native CTC aligner. It NEVER asserts rights were verified by software,
never scrapes/downloads recordings and never invents speaker or emotion labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unicodedata

import numpy as np
import soundfile as sf
import soxr

from .symbols import TOKENIZERS, unknown_characters

SAMPLE_RATE = 22050

# Metadata reflects the publishers' pages checked on 2026-10-26.
# Dataset-specific access terms and model-release obligations require human review.
SOURCES = {
    "mozilla_common_voice_vi_27": {
        "license": "CC0-1.0",
        "source_url": "https://datacollective.mozillafoundation.org/",
        "terms_url": "https://commonvoice.mozilla.org/en/datasets",
        "attribution": "Mozilla Common Voice Scripted Speech 27.0 (Vietnamese)",
        "purpose": "pronunciation_and_speaker_diversity",
        "access_note": "Respect dataset download terms; do not re-host clips or re-identify contributors.",
    },
    "fpt_fosd_v4": {
        "license": "CC-BY-4.0 + source-embedded FPT permission/notice",
        "source_url": "https://data.mendeley.com/datasets/k9sxg2twv4/4",
        "terms_url": "https://data.mendeley.com/datasets/k9sxg2twv4/4",
        "attribution": "FPT Corporation (2018); FPT Open Speech Dataset, via Duc Chung Tran (2020)",
        "purpose": "pronunciation_and_alignment",
        "access_note": "Preserve attribution and FPT source notice; commercial/model rights require review.",
    },
    "google_fleurs_vi": {
        "license": "CC-BY-4.0",
        "source_url": "https://huggingface.co/datasets/google/fleurs",
        "terms_url": "https://huggingface.co/datasets/google/fleurs",
        "attribution": "Google FLEURS corpus (Conneau et al., 2022), Vietnamese subset",
        "purpose": "held_out_evaluation_preferred",
        "access_note": "Attribute dataset and paper. Do not mix official test data into training.",
    },
    "actor_recordings": {
        "license": "custom-written-voice-and-training-consent",
        "source_url": "local://licensed-actor-recordings",
        "terms_url": "local://recording-consent",
        "attribution": "Actor recordings with documented consent supplied by dataset owner",
        "purpose": "expressive_dubbing_after_human_annotation",
        "access_note": "Keep signed performer agreements and usage permissions outside the public repo.",
    },
}


class IntakeError(ValueError):
    """Corpus import or provenance validation failure."""


def _safe_source_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not relative.strip():
        raise IntakeError("Audio path must be nonempty and relative")
    full = (root / candidate).resolve()
    if not full.is_relative_to(root.resolve()) or not full.is_file():
        raise IntakeError("Audio missing or outside authorized audio root")
    return full


def _wav_samples(path: Path) -> tuple[np.ndarray, int]:
    """Read soundfile-compatible audio, or safely decode MP3 using ffmpeg."""
    try:
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        return np.asarray(samples, dtype=np.float32), rate
    except (sf.LibsndfileError, RuntimeError):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise IntakeError("Cannot decode audio; install ffmpeg for MP3") from None
        # The source is a local, authorized file. No shell or remote protocols.
        with tempfile.TemporaryDirectory(prefix="vietdub_ingest_") as tmp:
            decoded = Path(tmp) / "decoded.wav"
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
                 "-i", str(path), "-vn", "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-c:a", "pcm_s16le", str(decoded)],
                capture_output=True, text=True, timeout=90, check=False,
            )
            if result.returncode or not decoded.is_file():
                raise IntakeError(f"ffmpeg decode failed: {result.stderr[:160]}")
            return sf.read(decoded, dtype="float32", always_2d=True)


def inspect_audio(path: Path, *, min_seconds: float = 0.4,
                  max_seconds: float = 20.0) -> tuple[np.ndarray, dict]:
    if min_seconds <= 0 or max_seconds <= min_seconds:
        raise IntakeError("Invalid duration limits")
    data, sample_rate = _wav_samples(path)
    if not 8000 <= sample_rate <= 192000 or data.ndim != 2 or not 1 <= data.shape[1] <= 2:
        raise IntakeError("Unsupported audio sample rate or channel layout")
    duration = data.shape[0] / sample_rate
    if not min_seconds <= duration <= max_seconds:
        raise IntakeError(f"Duration {duration:.3f}s outside accepted range")
    if not np.isfinite(data).all():
        raise IntakeError("NaN/Inf audio samples")
    mono = np.asarray(data.mean(axis=1), dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        mono = np.asarray(soxr.resample(mono, sample_rate, SAMPLE_RATE), dtype=np.float32)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    clipped_fraction = float(np.mean(np.abs(mono) >= 0.999))
    # These simple gates do NOT detect background noise, music, or transcript errors.
    if peak <= 0.001 or rms < 0.002:
        raise IntakeError("Silent / near-silent audio")
    if clipped_fraction > 0.005:
        raise IntakeError(f"Clipped audio fraction {clipped_fraction:.3%}")
    return mono, {"duration_sec": round(duration, 4), "peak": round(peak, 5),
                  "rms": round(rms, 6), "clipped_fraction": round(clipped_fraction, 6),
                  "original_sample_rate": sample_rate}


def _split_for_speaker(speaker_id: str | None, origin_split: str) -> str:
    if origin_split not in ("auto", "train", "dev", "test"):
        raise IntakeError("Unknown split")
    if origin_split != "auto":
        return origin_split
    if speaker_id is None:
        return "review_unassigned"
    # Split by speaker, not clip. All clips sharing an ID get the same split.
    slot = int(hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if slot < 80 else "dev" if slot < 90 else "test"


def _iter_rows(metadata: Path, corpus_format: str, *, path_column: str, text_column: str,
               speaker_column: str) -> list[dict]:
    if corpus_format not in ("common_voice", "table"):
        raise IntakeError("Unsupported metadata format")
    delimiter = "\t" if corpus_format == "common_voice" or metadata.suffix.lower() == ".tsv" else ","
    with metadata.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if not reader.fieldnames or not {path_column, text_column}.issubset(reader.fieldnames):
            raise IntakeError(f"Missing columns: {path_column}, {text_column}")
        return [dict(row) for row in reader]


def import_corpus(*, source: str, corpus_format: str, metadata: str | Path,
                  audio_root: str | Path, output_dir: str | Path,
                  rights_acknowledged: bool, terms_acknowledged: bool,
                  speaker_salt: str, tokenizer: str = "grapheme",
                  split: str = "auto", path_column: str | None = None,
                  text_column: str | None = None, speaker_column: str | None = None,
                  min_seconds: float = 0.4, max_seconds: float = 20.0) -> dict:
    if source not in SOURCES:
        raise IntakeError(f"Unknown dataset source: {source}")
    if not rights_acknowledged or not terms_acknowledged:
        raise IntakeError("Confirm rights and dataset-specific terms before import")
    if not isinstance(speaker_salt, str) or len(speaker_salt) < 12:
        raise IntakeError("Supply a private, random --speaker-salt of >=12 characters")
    if tokenizer not in TOKENIZERS:
        raise IntakeError("Unknown tokenizer")
    root, metadata_path, output = Path(audio_root).resolve(), Path(metadata).resolve(), Path(output_dir).resolve()
    if output == root or output.is_relative_to(root) or metadata_path.is_relative_to(output):
        raise IntakeError("Output directory must be separate from source recordings")
    if not root.is_dir() or not metadata_path.is_file():
        raise IntakeError("Metadata or audio directory not found")
    if corpus_format == "common_voice" and source != "mozilla_common_voice_vi_27":
        raise IntakeError("Common Voice parser requires the Common Voice source ID")
    if source == "google_fleurs_vi" and split == "auto":
        split = "test"  # preserve as evaluation by default; never auto-assign its samples to training
    path_column = path_column or ("path" if corpus_format == "common_voice" else "audio_path")
    text_column = text_column or ("sentence" if corpus_format == "common_voice" else "text")
    speaker_column = speaker_column or ("client_id" if corpus_format == "common_voice" else "speaker_id")
    rows = _iter_rows(metadata_path, corpus_format, path_column=path_column,
                      text_column=text_column, speaker_column=speaker_column)
    if not rows:
        raise IntakeError("Empty input metadata")
    accepted, rejected = [], []
    seen_audio: dict[str, str] = {}
    # Prepare in a private staging directory, then atomically promote the output.
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    audio_dir.mkdir(exist_ok=True)
    for index, row in enumerate(rows, 2):
        try:
            rel_path = (row.get(path_column) or "").strip()
            audio_path = _safe_source_path(root, rel_path)
            text = unicodedata.normalize("NFC", (row.get(text_column) or "").strip())
            if not text:
                raise IntakeError("Transcript is missing")
            unsupported = unknown_characters(text, tokenizer)
            if unsupported:
                raise IntakeError(f"Unsupported text tokens: {unsupported[:8]}")
            # Never use original speaker IDs in the exported manifest.
            raw_id = (row.get(speaker_column) or "").strip()
            speaker_id = ("cv_" if source.startswith("mozilla_") else "spk_") + hashlib.sha256(
                (speaker_salt + "|" + source + "|" + raw_id).encode("utf-8")
            ).hexdigest()[:20] if raw_id else None
            source_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
            if source_hash in seen_audio:
                raise IntakeError(f"Duplicate source audio (row {seen_audio[source_hash]})")
            mono, metrics = inspect_audio(audio_path, min_seconds=min_seconds,
                                          max_seconds=max_seconds)
            output_name = source_hash[:24] + ".wav"
            written = audio_dir / output_name
            sf.write(str(written), mono, SAMPLE_RATE, subtype="PCM_16")
            # Speaker ID is required by current aligner; 'unassigned' isn't a voice identity.
            entry = {
                "text": text,
                "audio_path": "audio/" + output_name,
                "speaker_id": speaker_id or "unassigned",
                "speaker_status": "pseudonymous_unverified" if raw_id else "unassigned",
                "emotion": "neutral", "emotion_status": "unverified_placeholder",
                "emotion_intensity": 0.5,
                "tokenizer": tokenizer,
                "alignment_verified": False, "alignment_status": "not_aligned",
                "dataset_source": source, "license": SOURCES[source]["license"],
                "source_url": SOURCES[source]["source_url"],
                "attribution": SOURCES[source]["attribution"],
                "source_sha256": source_hash,
                "split": _split_for_speaker(speaker_id, split),
                "audio_quality": metrics,
            }
            accepted.append(entry)
            seen_audio[source_hash] = str(index)
        except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            rejected.append({"row": index, "reason": str(exc)[:240]})
    if not accepted:
        raise IntakeError(f"No valid recordings. Rejected: {rejected[:5]}")
    def write_jsonl(target: Path, records: list[dict]) -> None:
        temp = target.with_suffix(target.suffix + ".part")
        try:
            temp.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                                    for item in records), encoding="utf-8")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    write_jsonl(output / "transcripts.jsonl", accepted)
    write_jsonl(output / "rejected.jsonl", rejected)
    summary = {
        "source": source, "license": SOURCES[source]["license"],
        "rights_and_terms_attested_by_operator": True,
        "license_and_consents_verified_by_software": False,
        "accepted": len(accepted), "rejected": len(rejected),
        "speakers_pseudonymous": len({r["speaker_id"] for r in accepted if r["speaker_id"] != "unassigned"}),
        "split_counts": {name: sum(r["split"] == name for r in accepted)
                         for name in ("train", "dev", "test", "review_unassigned")},
        "duration_hours": round(sum(r["audio_quality"]["duration_sec"] for r in accepted) / 3600, 5),
        "dataset_status": "UNALIGNED_AND_EMOTION_UNVERIFIED",
        "training_authorization": "CTC_ALIGNMENT_ONLY_UNTIL_HUMAN_REVIEW",
    }
    (output / "intake_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vietdub-datasets", description="Import licensed LOCAL Vietnamese audio, not download it")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources")
    ingest = sub.add_parser("import")
    ingest.add_argument("--source", required=True, choices=sorted(SOURCES))
    ingest.add_argument("--format", dest="corpus_format", choices=("common_voice", "table"), required=True)
    ingest.add_argument("--metadata", required=True)
    ingest.add_argument("--audio-root", required=True)
    ingest.add_argument("--output-dir", required=True)
    ingest.add_argument("--tokenizer", choices=TOKENIZERS, default="grapheme")
    ingest.add_argument("--split", choices=("auto", "train", "dev", "test"), default="auto")
    salt_group = ingest.add_mutually_exclusive_group(required=True)
    salt_group.add_argument("--speaker-salt")
    salt_group.add_argument("--speaker-salt-env", metavar="ENV_NAME",
                            help="Read salt from an environment variable, avoiding process-argument exposure")
    ingest.add_argument("--path-column")
    ingest.add_argument("--text-column")
    ingest.add_argument("--speaker-column")
    ingest.add_argument("--acknowledge-rights", action="store_true")
    ingest.add_argument("--acknowledge-terms", action="store_true")
    ingest.add_argument("--min-seconds", type=float, default=0.4)
    ingest.add_argument("--max-seconds", type=float, default=20)
    args = parser.parse_args(argv)
    if args.command == "sources":
        print(json.dumps(SOURCES, ensure_ascii=False, indent=2))
    else:
        kwargs = vars(args).copy()
        del kwargs["command"]
        kwargs["rights_acknowledged"] = kwargs.pop("acknowledge_rights")
        kwargs["terms_acknowledged"] = kwargs.pop("acknowledge_terms")
        env_name = kwargs.pop("speaker_salt_env")
        if env_name:
            kwargs["speaker_salt"] = os.environ.get(env_name, "")
        print(json.dumps(import_corpus(**kwargs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
