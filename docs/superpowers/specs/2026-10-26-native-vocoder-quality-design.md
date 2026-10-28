# VietDub Native Vocoder Quality — Design

## Intent and acceptance criterion

VietDub must eventually **speak a held-out Vietnamese sentence intelligibly** with independently trained weights. Existing research outputs are ~0.43-second noisy waveforms and do not qualify. The first useful, independent milestone is a capable vocoder that reconstructs held-out speech audio from mel features before integrating a text-to-mel acoustic model. No CI green check, RMS statistic, or valid WAV proves speech intelligibility.

## Scope of this change (phase 1, independent and testable)

1. Separate vocoder training from phoneme alignment and speaker/emotion annotation. A vocoder learns `mel -> waveform`, so it should not be blocked by token durations.
2. Accept only an **operator-approved, rights-reviewed local recording manifest** with separate train/dev clips; enforce provenance, path boundaries, sample format and duplicate/holdout checks. These fields are assertions, not independent legal verification.
3. Introduce a **new trainable HiFi-GAN-inspired generator** with period discriminators, multiresolution spectral training and feature matching. Do not import pretrained TTS models, voices or checkpoints.
4. Produce private generator checkpoint, JSON metrics and a **held-out reconstructed WAV**; never copy source voice files into public artifacts. Do not claim audible quality until side-by-side human review.
5. Keep current acoustic and legacy research paths unchanged. A later phase can implement a phoneme/VITS-alignment acoustic model once reconstruction quality is established.

## Approaches considered

- **A. Increase steps for existing tiny convolution vocoder:** least code, but insufficient architecture/corpus and recent 100-step proof-of-noise is not compelling; rejected.
- **B. Plug in a pretrained external vocoder:** fastest audible demo, but violates independently trained VietDub model objective; rejected.
- **C. Isolated native HiFi-GAN-inspired pipeline (chosen):** more implementation and GPU training, but delivers an independently trainable reconstruction subsystem with testable train/dev fidelity and preserves existing APIs.

## Module contracts

- `vocoder_corpus.read_vocoder_manifest(path, split='train') -> list[dict]` validates **all** rows (including split leakage), then selects train/dev; no transcript/alignment required. Mandatory fields: `audio_path`, `dataset_source`, `license`, `source_url`, `attribution`, `split`, `rights_verified=true`, `vocoder_training_approved=true`. Actor recordings additionally require `voice_training_consent_verified=true`. Unapproved FOSD probe rows cannot pass the gate.
- `vocoder_corpus.VocoderWaveDataset(manifest, cfg, split, crop_frames)` returns `(log_mel[n_mels,frames], waveform[frames*hop])` from identical aligned crops.
- `hifigan.HiFiGANGenerator(cfg, channels=128)` upscales by `8*8*4=256`, giving exactly `frames*256` samples. `MultiPeriodDiscriminator` uses periods `(2,3,5,7,11)`.
- `hifigan_train.train_hifigan(manifest, output_dir, steps, ...) -> dict` trains only on `train`, evaluates on `dev`, saves checkpoint as private, and records a reconstruction metric without misleading intelligibility claims.

## Risks and guardrails

- Audio files may be licensed for evaluation but not training; no automated attestation can prove rights. Human review and use restrictions are required.
- Source hashes alone are untrusted; check actual prepared audio byte hashes to reject duplicate splits.
- Short clips and distorted/unclear speech will prevent adequate reconstruction; CPU unit tests using synthetic sine waves do **not** establish speech quality.
- Generator is "HiFi-GAN-inspired" rather than a claim of exact reference reproduction. GPU-scale training, stronger multiscale discriminators, data loading throughput, validation by human listening and architecture optimization remain future work.
- Acoustic/phonetic alignment, emotion conditioning and the movie dubbing integration are explicitly **out of scope for this PR**.

## Acceptance and release policy

- Locally verify red/green tests, `python -m compileall`, and full CI on GitHub before marking the PR review-ready.
- A reconstructed held-out waveform must be listened to against the licensed source, with explicit permission to share recordings; otherwise publish only non-identifying metrics.
- The user-facing definition of VietDub TTS working is a full, unseen Vietnamese sentence understood by listeners; this PR does not claim to meet it.
