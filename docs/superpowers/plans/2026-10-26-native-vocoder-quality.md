# Native Vocoder Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a genuinely trainable, independently versioned neural vocoder with a licensed train/dev dataset and reconstruction evidence rather than a noise-demo-only pipeline.

**Architecture:** Separate rights-gated `VocoderWaveDataset` from acoustic alignment, implement an isolated generator + multi-period critic, then add GPU-capable training, private checkpoint and held-out validation. Existing acoustic checkpoints/API remain unaffected.

**Tech Stack:** Python 3.11+, PyTorch, torchaudio, soundfile, numpy, pytest; optional CUDA.

**Spec:** `docs/superpowers/specs/2026-10-26-native-vocoder-quality-design.md`

## Global Constraints

- Never mark `training_approved`, `alignment_verified`, emotion, transcript or voice rights as true automatically.
- No pretrained speech generator or checkpoint dependency.
- Waveform training data remain local; CI uses synthetic fixtures only.
- Separate branch and PR, never commit directly to `main`.
- Evidence must include held-out reconstruction and explicit `intelligibility_verified: false` until listened to and accepted.

## Review Focus

1. Identical audio in train/dev despite different asserted hashes -> reject (Task 1).
2. WAV path traversal / symlink -> reject (Task 1).
3. Missing training rights, performer consent for actor data -> reject (Task 1).
4. Upsampling frame-length mismatch -> fail loudly (Task 2).
5. Training mistakenly on dev and publishing original voices -> exclude dev optimizer and source audio from artifacts (Task 3).

---

### Task 1: Independent approved vocoder corpus

**Files:** `src/vietdub_native/vocoder_corpus.py`, `tests/test_hifigan_training.py`

**Interfaces:** `read_vocoder_manifest(path, split)`, `VocoderWaveDataset(manifest, cfg, split, crop_frames)`.

- [x] **RED:** Write corpus tests with no transcript or alignment; test missing approval, traversal, source duplicates and actor consent. Run `PYTHONPATH=src pytest tests/test_hifigan_training.py -q` and observe missing module / failing duplicate guard.
- [x] **GREEN:** Validate each manifest row and compute deterministic matching mel/waveform crops; check SHA of prepared file, not merely manifest assertion.
- [x] **VERIFY:** `PYTHONPATH=src pytest tests/test_hifigan_training.py -q` locally.
- [ ] **COMMIT/REVIEW:** Include with atomic PR and rerun repository-wide tests in GitHub Actions.

### Task 2: Native HiFi-GAN generator and discriminator

**Files:** `src/vietdub_native/hifigan.py`, `tests/test_hifigan_training.py`

**Interfaces:** `HiFiGANGenerator(cfg, channels)`, `MultiPeriodDiscriminator(channels, periods)`, `discriminator_loss`, `generator_loss`.

- [x] **RED:** Test output exact length, finite loss and input gradients before implementation.
- [x] **GREEN:** Implement 8x8x4 transposed upsampling, dilated residual blocks and period critics.
- [x] **VERIFY:** Assert `mel[2,16,12] -> audio[2,3072]` and backpropagation on CPU.
- [ ] **COMMIT/REVIEW:** Review gradient flow, long-audio inference and architectural simplifications.

### Task 3: Independent vocoder training and evaluation

**Files:** `src/vietdub_native/hifigan_train.py`, `tests/test_hifigan_training.py`, `docs/NATIVE_VOCODER_TRAINING.md`

**Interfaces:** `train_hifigan(manifest, output_dir, *, cfg, steps, crop_frames, generator_channels, discriminator_channels, lr, device, seed) -> dict`.

- [x] **RED:** Test one-step synthetic train/dev run and expect checkpoint, report and reconstructed (not original) dev WAV.
- [x] **GREEN:** Alternate discriminator/generator updates, use spectral and feature-matching losses, save own generator weights and held-out reconstruction.
- [x] **VERIFY:** Check `intelligibility_verified=false`; run synthetic fixtures locally.
- [ ] **COMMIT/REVIEW:** CI complete; do not distribute training/dev recordings.

### Task 4: GPU-scale real-data evaluation (next independent milestone)

**Files:** `.github/workflows/...` (optional manual workflow), dataset-specific review manifest and private GPU runner config.

- [ ] Acquire and verify sufficient rights-cleared Vietnamese speech, review manifests and prepare separate dev set.
- [ ] Train generator/discriminator with resumable epochs/steps and performance budget on actual GPU.
- [ ] Compare reconstructed held-out speech to source via listening and objective metrics; document results.
- [ ] Only when acceptable reconstruction is proven, begin VITS/text-to-mel design and training.

## Current Execution Evidence

Initial local TDD produced an import error (`vocoder_corpus` absent), then 7 passing tests. Added duplicate-content test failed as expected, fixed actual byte-hash duplicate detection, then 9 local tests passed. GitHub Actions must independently confirm the full suite before PR completion.


### Task 5: Resumable GAN training and audible-length held-out evaluation (2026-10-27)

**Files:** `src/vietdub_native/vocoder_corpus.py`, `src/vietdub_native/hifigan_train.py`, `tests/test_hifigan_training.py`, `docs/NATIVE_VOCODER_TRAINING.md`

**Interfaces:** `VocoderWaveDataset.sample_at(index, *, step) -> (mel, waveform)` and `train_hifigan(..., resume=False, checkpoint_every=250, eval_seconds=5.0) -> dict`.

- [x] **RED:** Added four failing tests: per-step crop variability with deterministic replay, one-shot versus resumed model-weight equivalence, rejection of changed manifest without clobbering checkpoint, and longer held-out dev output. Observed all four fail locally for the intended missing behavior.
- [x] **GREEN:** Deterministic step-dependent crops; persisted full optimizer+discriminator state; exact corpus and config signature validation; atomic checkpoint replacement; longer held-out dev reconstruction.
- [x] **VERIFY:** `PYTHONPATH=src OMP_NUM_THREADS=2 pytest -q tests/test_hifigan_training.py` -> 13 passed on CPU local. Full local checkout is incomplete (earlier source bundles lack latest `symbols.py` and `text_frontend.py`), so GitHub Actions CI is the authoritative full-suite check.
- [ ] **REVIEW:** Investigate GPU resume, dataset throughput and real approved-corpus listening quality. No claims of intelligible speech until independently listened to.


### Task 6: Offline speaker-disjoint Common Voice importer (2026-10-27)

**Files:** `src/vietdub_native/commonvoice_vocoder.py`, `tests/test_commonvoice_vocoder.py`, `docs/NATIVE_VOCODER_TRAINING.md`, `pyproject.toml`

**Interfaces:** `prepare_commonvoice(source_root, output_dir, *, max_clips=10000, seed=7, dev_fraction=0.1, rights_reviewed=False, training_approved=False) -> dict`.

- [x] **RED:** Import of `vietdub_native.commonvoice_vocoder` initially failed; four tests describe review-only defaults, rights-approved speaker-disjoint manifests, input path traversal protection, and minimum speaker count.
- [x] **GREEN:** Implement local MP3-to-WAV importer with per-speaker deterministic partitioning; do not infer rights from dataset metadata.
- [x] **VERIFY:** Added an end-to-end fixture test from synthetic MP3 to one-step vocoder train. `PYTHONPATH=src OMP_NUM_THREADS=2 pytest -q tests/test_commonvoice_vocoder.py tests/test_hifigan_training.py` -> 18 passed locally.
- [ ] **REVIEW:** On authorized actual Common Voice Vietnamese release, inspect speaker metadata availability, quality rejection rate, GPU utilization and attribution needs. CI fixtures are *not* actual Vietnamese speech.


### Task 7: Reject near-silent data and provide a GPU execution handoff (2026-10-27)

**Files:** `src/vietdub_native/commonvoice_vocoder.py`, `tests/test_commonvoice_vocoder.py`, `notebooks/kaggle_vocoder_gpu.ipynb`, `docs/NATIVE_VOCODER_TRAINING.md`

- [x] **RED:** Test with near-silent MP3 expects rejected clip and `rejected_low_quality=1`; observed one failure (8 accepted instead of 7).
- [x] **GREEN:** Inspect decoded WAV for silence/clipping before manifest admission. Record dropped quality/decode clip counts.
- [x] **VERIFY:** `PYTHONPATH=src OMP_NUM_THREADS=2 pytest -q tests/test_commonvoice_vocoder.py tests/test_hifigan_training.py` -> 19 passed locally. Generated Kaggle notebook with approval checks initially disabled; syntax-validated code cells.
- [ ] **REVIEW:** GPU run and audible held-out reconstruction are NOT verified. Kaggle notebook requires attached authorized corpus, operator license review and GPU enablement; never claim the generated audio works without listening.


### Task 8: Superpowers review fix — train-speaker diversity (2026-10-27)

**Files:** `src/vietdub_native/commonvoice_vocoder.py`, `tests/test_commonvoice_vocoder.py`

- [x] **RED:** Fixed-seed minimal corpus (`seed=4`, 3 speakers, `max_clips=3`) used only one distinct train speaker; observed assertion failure: expected 2, got 1.
- [x] **GREEN:** Reserve one recording each from two distinct train speakers and one dev speaker before filling bounded slots.
- [x] **VERIFY:** `PYTHONPATH=src OMP_NUM_THREADS=2 pytest -q tests/test_commonvoice_vocoder.py tests/test_hifigan_training.py` -> 20 passed on CPU local.
- [ ] **REVIEW:** Real Common Voice `client_id` values and speaker consistency remain external facts; user must verify source and complete an actual GPU training run.
