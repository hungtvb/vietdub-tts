# Adam Synthetic Corpus Intake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely prepare Phitran21/Vietnamese-Adam-TTS-50h as an optional, rights-attested source of mel/audio reconstruction data for the existing native vocoder.

**Architecture:** A ZIP-streaming preparer validates id/text/WAV pairs, resamples selected samples and writes separately gated vocoder and review-only alignment manifests. A Kaggle notebook mounts the archive as input; trainer and checkpoint code remain unchanged.

**Tech Stack:** Python 3.10+, zipfile, soundfile, numpy, soxr, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-adam-synthetic-corpus-intake-design.md`

## Global Constraints

- Training gate disabled by default; source license label and upstream synthetic voice asset rights are independent.
- Do not extract entire 17.5 GB ZIP; no network download in importer; no source recordings/checkpoints on GitHub.
- One voice preset → utterance-group-disjoint evaluation, never speaker-disjoint claims.
- Do not modify legacy acoustic/alignment or Common Voice training code.

## Review Focus

- Malicious ZIP member traversal → reject (Task 1).
- Repeated source IDs / identical normalized text across splits → reject or group (Task 1).
- Missing, malformed, silent, clipped or wrong-sample-rate WAV → skip with counts (Task 1).
- Corpus rights flag true but upstream voice rights unknown → refuse training (Task 2).
- CLI and GPU examples must not silently enable training or claim speech intelligibility (Task 3).

---

### Task 1 — Offline ZIP audio/text importer

**Files:** `src/vietdub_native/adam_tts_vocoder.py`, `tests/test_adam_tts_vocoder.py`

**Interfaces:** `prepare_adam_tts(archive_path, output_dir, *, max_clips=5000, seed=7, dev_fraction=0.05, rights_reviewed=False, upstream_voice_rights_reviewed=False, training_approved=False) -> dict`.

- [x] Write failing tests for review-only defaults, source quality, unsafe ZIP and text-group leakage.
- [x] Run `PYTHONPATH=src pytest tests/test_adam_tts_vocoder.py -q` → expected missing module RED.
- [x] Implement bounded ZIP parser, sample resampling, SHA checks and disjoint text grouping.
- [x] Run test again → GREEN; 7 passed before integration enhancements.

### Task 2 — Rights gate and existing trainer integration

**Files:** `src/vietdub_native/adam_tts_vocoder.py`, `tests/test_adam_tts_vocoder.py`

- [x] Test rejects upstream voice-rights flag missing despite dataset license acknowledgment.
- [x] Test approved synthetic waveform fixtures pass `read_vocoder_manifest` and `train_hifigan(..., steps=1)`.
- [x] Write RED test for separate transcript-only review output, then implement `transcripts_review.jsonl` without fabricated verification.
- [x] Run targeted tests → GREEN.

### Task 3 — CLI, docs, GPU notebook and review

**Files:** `pyproject.toml`, `docs/ADAM_TTS_50H.md`, `notebooks/kaggle_adam_tts_gpu.ipynb`, this plan.

- [x] Add `vietdub-adam-vocoder` entrypoint.
- [x] Add a review-gated Kaggle notebook that mounts an externally supplied ZIP and trains only after human confirmation.
- [ ] Run `python -m compileall -q src tests`, entire `pytest -q`, and validate notebook JSON and commands.
- [ ] Push to existing draft PR #4, inspect full GitHub CI and complete PR code review.
- [ ] Do **not** claim audible Adam reconstruction without a genuine rights-cleared GPU training/listening report.
