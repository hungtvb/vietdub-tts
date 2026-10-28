# Adam Remote Source Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inspect actual Adam ZIP metadata and <=3 WAVs with <=16 MiB HTTP transfer.

**Architecture:** Seekable HTTP Range adapter + Python zipfile, reusing Adam importer validation.

**Tech Stack:** Python stdlib urllib, ZIP, NumPy, SoundFile, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-28-adam-remote-source-probe-design.md`

## Global Constraints
- Never bypass upstream synthetic voice rights gate or upload recordings.
- Cap per-run transfer at 16 MiB and sample count at 3.
- No pretrained TTS checkpoint.

## Review Focus
- Server ignores Range: reject (Task 1).
- Content-Range mismatch: reject (Task 1).
- ZIP path traversal/duplicate entries, metadata oversized: reject via shared importer validator (Task 2).
- WAV missing/corrupt: report rejection (Task 2).
- HF unavailable: fail without pretending the probe succeeded (Task 3).

### Task 1: HTTP Range reader
**Files:** Create `src/vietdub_native/adam_remote_probe.py`, test `tests/test_adam_remote_probe.py`.
- [x] Write failing tests for seek/cache and full-GET rejection.
- [x] Verify RED: `ModuleNotFoundError` for missing `adam_remote_probe`.
- [x] Implement reader with bounded download and cache.
- [x] Verify GREEN: local pytest tests pass.

### Task 2: Remote ZIP inspect
**Files:** Same module/test.
- [x] Add local 48-kHz synthetic WAV ZIP fixture; check metadata and >=2 WAVs.
- [x] Probe sparse WAVs and generate aggregate report.
- [x] Test limits.

### Task 3: External evidence
**Files:** Add `.github/workflows/adam-source-probe.yml` and `docs/ADAM_REMOTE_PROBE.md`.
- [x] Create bounded real probe workflow; publish only report.
- [ ] Push to feature branch and inspect remote outcome.
- [ ] Document exact findings or remote failure.
- [ ] Review PR and keep draft until real rights/training resolved.
