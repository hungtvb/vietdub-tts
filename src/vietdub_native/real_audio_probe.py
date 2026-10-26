"""Bounded FOSD *real audio* research probe via public HF Dataset Viewer.

Intentionally NOT a training data approval or an identity/emotion annotator.
No audio is uploaded, committed, or released by the GitHub workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib import parse, request

import soundfile as sf

from .data_sources import inspect_audio, IntakeError
from .symbols import unknown_characters

DATASET = "awghuku/fosd"  # unofficial mirror; publisher original at Mendeley
SERVER = "https://datasets-server.huggingface.co"
MAX_BYTES = 3 * 1024 * 1024
FPT_URL = "https://data.mendeley.com/datasets/k9sxg2twv4/4"


class ProbeError(ValueError):
    pass


def _read(url: str, size_limit: int = MAX_BYTES) -> bytes:
    req = request.Request(url, headers={"User-Agent": "VietDub-research-probe/1.0"})
    with request.urlopen(req, timeout=25) as response:
        payload = response.read(size_limit + 1)
    if len(payload) > size_limit:
        raise ProbeError(f"Download exceeded {size_limit} bytes")
    return payload


def _get_json(url: str) -> dict:
    payload = _read(url, 512 * 1024)
    try:
        obj = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProbeError("Dataset server did not respond with JSON") from exc
    if not isinstance(obj, dict):
        raise ProbeError("Invalid dataset server payload")
    return obj


def _audio_url(value: object) -> str:
    # HF Dataset Viewer wraps even a single recording in an audio list.
    # Multi-clip rows require transcript review: never guess a matching clip.
    if isinstance(value, list):
        if len(value) != 1:
            raise ProbeError("Ambiguous audio segments: expected exactly one audio per transcript")
        value = value[0]
    if not isinstance(value, dict) or not isinstance(value.get("src"), str):
        raise ProbeError("Dataset row has no audio.src URL")
    src = value["src"]
    if src.startswith("/") and not src.startswith("//"):
        src = SERVER + src
    parsed = parse.urlsplit(src)
    # Viewer-provided URLs are trusted only for the supported HF domains.
    if parsed.scheme != "https" or parsed.hostname not in (
        "datasets-server.huggingface.co", "huggingface.co", "cdn-lfs.huggingface.co"
    ) or parsed.username or parsed.password:
        raise ProbeError("Unexpected audio host from dataset server")
    return src


def _find_split() -> tuple[str, str]:
    query = parse.urlencode({"dataset": DATASET})
    payload = _get_json(f"{SERVER}/splits?{query}")
    matches = [r for r in payload.get("splits", [])
               if isinstance(r, dict) and r.get("dataset") == DATASET
               and r.get("split") == "train"]
    if not matches:
        raise ProbeError("Public train split not offered by Dataset Viewer")
    return str(matches[0]["config"]), "train"


def _load_rows(config: str, split: str, count: int) -> list[dict]:
    query = parse.urlencode({"dataset": DATASET, "config": config,
                             "split": split, "offset": 0, "length": count})
    payload = _get_json(f"{SERVER}/rows?{query}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ProbeError("Dataset Viewer did not return any rows")
    return rows


def _to_wav(audio_bytes: bytes, destination: Path) -> None:
    if not audio_bytes:
        raise ProbeError("Empty audio response")
    with tempfile.TemporaryDirectory(prefix="vietdub_real_audio_") as td:
        raw = Path(td) / "clip.bin"
        raw.write_bytes(audio_bytes)
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(raw),
             "-vn", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", "-y", str(destination)],
            capture_output=True, text=True, timeout=30, check=False,
        )
    if result.returncode or not destination.is_file():
        raise ProbeError(f"ffmpeg could not decode clip: {result.stderr[:150]}")


def probe(output_dir: str | Path, *, rights_acknowledged: bool,
          limit: int = 5, min_ok: int = 2) -> dict:
    """Download at most N *public* dataset-viewer samples and measure real audio.

    No speaker/emotion/transcript approval is inferred. Outputs review-only files.
    """
    if not rights_acknowledged:
        raise ProbeError("Review FPT FOSD license, source notice, and mirror provenance first")
    if not 1 <= limit <= 12 or not 1 <= min_ok <= limit:
        raise ProbeError("Limit 1..12; require 1 <= min_ok <= limit")
    dest = Path(output_dir).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise ProbeError("Output directory is not empty; refusing to mix runs")
    dest.mkdir(parents=True, exist_ok=True)
    clips = dest / "audio"
    clips.mkdir()
    config, split = _find_split()
    entries, rejected = [], []
    rows = _load_rows(config, split, limit)
    for item in rows:
        index = item.get("row_idx", "unknown") if isinstance(item, dict) else "unknown"
        try:
            if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                raise ProbeError("Malformed dataset row")
            row = item["row"]
            utterance = row.get("text")
            if not isinstance(utterance, str) or not utterance.strip():
                raise ProbeError("Missing transcript")
            unknown = unknown_characters(utterance, "grapheme")
            if unknown:
                raise ProbeError(f"Unsupported transcript chars: {unknown[:8]}")
            raw = _read(_audio_url(row.get("audio")))
            sha = hashlib.sha256(raw).hexdigest()
            out_file = clips / f"{sha[:20]}.wav"
            _to_wav(raw, out_file)
            try:
                _, quality = inspect_audio(out_file, min_seconds=0.4, max_seconds=20)
                if sf.info(out_file).samplerate != 22050:
                    raise ProbeError("Output WAV sample rate mismatch")
            except Exception:
                out_file.unlink(missing_ok=True)
                raise
            entries.append({"text": utterance.strip(), "audio_path": str(Path("audio") / out_file.name),
                            "speaker_id": "unassigned", "speaker_status": "unassigned",
                            "emotion": "neutral", "emotion_status": "unverified_placeholder",
                            "emotion_intensity": 0.5, "tokenizer": "grapheme",
                            "dataset_source": "fpt_fosd_v4", "dataset_mirror": DATASET,
                            "split": "review_unassigned", "transcript_verified": False,
                            "alignment_verified": False, "license": "CC-BY-4.0 + FPT notice",
                            "source_url": FPT_URL, "original_dataset_row": index,
                            "source_sha256": sha, "audio_quality": quality})
        except (ProbeError, IntakeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
            rejected.append({"row_idx": index, "reason": str(exc)[:220]})
    report = {"source_dataset": DATASET, "original_publisher": FPT_URL,
              "dataset_config": config, "source_split": split,
              "downloaded_decoded": len(entries), "rejected": len(rejected),
              "total_seconds": round(sum(x["audio_quality"]["duration_sec"] for x in entries), 4),
              "training_approved": False,
              "note": "Real source waveforms inspected; unverified speaker, transcript, emotion and alignment. No trained TTS checkpoint."}
    (dest / "transcripts.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                       for row in entries), encoding="utf-8")
    (dest / "rejects.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(entries) < min_ok:
        raise ProbeError(f"Only {len(entries)} valid real recordings; need {min_ok}; see report.json")
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="FOSD real audio public mirror, bounded validation probe")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--min-ok", type=int, default=2)
    p.add_argument("--acknowledge-rights", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(probe(args.output, rights_acknowledged=args.acknowledge_rights,
                           limit=args.limit, min_ok=args.min_ok), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
