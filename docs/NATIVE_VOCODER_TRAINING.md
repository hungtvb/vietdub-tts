# Native HiFi-GAN-inspired vocoder — experimental training

This is an **independently initialized waveform generator**, not VieNeu, a pretrained checkpoint, or a guaranteed intelligible TTS model. It only reconstructs audio from mel-spectrograms. Text-to-speech will require a separately trained acoustic model.

## Input manifest

Create a **locally verified** `vocoder.jsonl` alongside mono 22050 Hz WAV files. Each row requires `audio_path`, `dataset_source`, `split` (`train`/`dev`), `license`, `source_url`, `attribution`, `rights_verified: true` and `vocoder_training_approved: true`. For source `actor_recordings`, also require `voice_training_consent_verified: true` following consent review.

This is a schema example only. Do **not** set approval fields to true until actual rights were reviewed. The original FOSD probe remains `review_unassigned` and fails this gate intentionally.

```json
{"audio_path":"audio/approved_001.wav","split":"train","dataset_source":"approved_recordings","license":"PER-RECORDING LICENSE REVIEW REQUIRED","source_url":"https://example.org/provenance","attribution":"RECORDING OWNER TO VERIFY","rights_verified":true,"vocoder_training_approved":true}
```

The trainer requires two or more approved train clips and at least one approved held-out dev clip. It does not need transcripts, token durations or emotion labels. It rejects duplicate source hashes and duplicate prepared WAV contents across train/dev, path escapes, non-mono audio or unexpected sample rates. `rights_verified` is an assertion, not proof of legal permission.

## Start training (GPU recommended)

```bash
pip install -e '.[dev]'
python -m vietdub_native.hifigan_train \
  --manifest /path/to/reviewed/vocoder.jsonl \
  --output-dir /path/to/private/runs/vocoder-v1 \
  --steps 100000 --crop-frames 48 --device cuda
```

The number of steps is illustrative, not a quality guarantee. The current trainer is a baseline and lacks true large-scale data loader throughput, resume/scheduler and formal quality benchmarks; these must be addressed before long production runs.

Outputs include `private_checkpoints/hifigan.pt`, `training_history.json`, `evaluation.json` and `dev_reconstructed.wav`. **Never distribute dev source recordings or checkpoints derived from unreviewed voices.** The output reconstructed WAV requires a listener to compare it with the licensed original. A finite GAN loss or the appearance of WAV is not proof of quality.

## Superpowers workflow

Apply the upstream [`obra/superpowers` using-superpowers](https://github.com/obra/superpowers/tree/main/skills/using-superpowers) with brainstorming/design -> writing-plans -> isolated branch -> TDD -> verification -> code review. Read the design and plan under `docs/superpowers/` before changing the feature. Do not merge merely because CI passes.
