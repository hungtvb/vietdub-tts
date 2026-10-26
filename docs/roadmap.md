# Roadmap and acceptance criteria

## P1 (current): independent engine MVP
- REST API and CLI, VieNeu v3 Turbo lazy adapter, character registry, 44.1kHz output, ZIP + manifest, input validation, small-ratio atempo, offline tests and GitHub Actions CI.
- Exit: all unit/API tests pass; human run confirms at least one audible clip via installed model; any missing GPU and model issues recorded. Do not claim audio evaluation from mock tests.

## P2: emotion direction and audio evidence
- Scene-context-aware emotion tags with LLM and reviewer feedback; explicit capabilities per backend; record suitable voice references and consent.
- Comparative emotional reference samples and multiple takes; forced ASR transcript QA, speaker similarity metrics, loudness/clipping checks.
- Blind human listening: compare neutral vs sad/angry/afraid/crying. No automatic success claims from model arguments.

## P3: voices and scene continuity
- Validated actor/character anchors per scene and per-character presets/reference library; clean licensed audio; voice stability score; identify overlap and cross-talk.
- Review UI in VietDub maps diarization labels to real characters before synth.

## P4: expressive acoustic model
- Build licensed, consented acted Vietnamese dataset with reliable transcripts, speaker labels, scene-level emotions and expressive prosody annotations.
- Establish baseline v3 checkpoint metrics, train LoRA, compare side-by-side with unseen film dialogues; consider architecture with explicit emotion conditioning when reference prompts aren't reliable.
- Track provenance, license, speaker consent and audio model version in the artifact manifest.

## P5: production service
- Queues, idempotency and hash-based cache (text+voice+model+emotion+timeline), cancellation, health/readiness, concurrency policy, model version pins, observability and authentication.
- E2E full-film VietDub workflows on varied sources (not overfitted to a single test clip).

## Validation dimensions
- Speaker similarity and stability across scenes (blind review, embedding similarity where possible).
- Pronunciation accuracy using forced alignment/ASR and manual audit of difficult Vietnamese text.
- Emotional perception: blind listener agreement, naturalness MOS, intelligibility and no unsupported cue artifacts.
- Absolute A/V timing and overlap handling; avoid speech truncation and waveform clipping.
- CPU/GPU real-time factor, peak RAM/VRAM and 95th percentile request latency.
