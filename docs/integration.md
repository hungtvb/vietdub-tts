# VietDub integration contract (Phase 1)

VietDub presently uses `tts/edge_tts.py`, `tts/duration_fit.py`, and `core/orchestrator.py`; all emit/consume per-segment mono 44.1kHz WAVs. This service lives in a separate repository; **no VietDub code is changed by this PR**.

1. During the review stage, associate every diarization `speaker_id` with a stable character voice in `VIETDUB_TTS_VOICES_FILE`. Do not auto-approve unverified speaker clustering.
2. Send translated `text_vi`, `index`, `start`, `end`, `speaker_id` and optional `emotion`, `emotion_intensity` to `/v1/dubbing/segments`.
3. Unzip returned audio to VietDub's per-job `stages/tts` directory, verify all files and durations; read `manifest.json` before setting `seg.audio_vi`.
4. Reuse VietDub's existing `duration_fit`/mix/render stages; account for the fact the returned segments already have timing padding. Keep overlapping spoken tracks and their timestamps intact.
5. On HTTP 422 for duration overflow, split or rewrite the translation, or flag for review. Never silently substitute stale WAV or treat missing lines as success.

The current implementation generates all segments synchronously and holds ZIP output in memory. Production phase requires an authenticated async job queue, progress events, disk-backed exports, durable caching and cancellation. Emotion metadata is *not* proof that a listener can hear that emotion: blind audio evaluations are required.
