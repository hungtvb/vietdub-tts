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

The number of steps is illustrative, not a quality guarantee. The trainer supports deterministic step-dependent crops and exact optimizer+discriminator resume from a private training-state checkpoint. It still lacks true large-scale DataLoader throughput, schedulers and validated quality benchmarks; do not assume production readiness.

To resume interrupted training with the **same manifest and audio bytes, configuration, crop size, optimizer hyperparameters and seed**, keep the original private output directory and run:

```bash
python -m vietdub_native.hifigan_train \
  --manifest /path/to/reviewed/vocoder.jsonl \
  --output-dir /path/to/private/runs/vocoder-v1 \
  --steps 120000 --crop-frames 48 --device cuda --resume \
  --checkpoint-every 250 --eval-seconds 5
```

`--steps` is the **total target optimizer steps**, not additional steps. Each training step uses a deterministically varied crop (replayable after resume). `training_state.pt` saves **both models, both AdamW optimizer states, their hyperparameters, torch RNG and loss history**; atomic checkpoint replacement helps preserve the prior checkpoint on an interrupted write. The resume code rejects any changed corpus manifest or audio content. Store checkpoint files **privately**, not in public CI artifacts. A resuming run must use the same device class and numeric environment for exactly reproducible weights.

The dev reconstruction uses up to `--eval-seconds` of a genuine held-out clip; it is **not** limited to the train crop length. This is a generated reconstruction of the original recording's mel features, **not text-to-speech**. `intelligibility_verified` remains `false` and listening against the original is required. To build a comparison, privately listen to the held-out original already present on your licensed machine; do not commit it or expose it in GitHub artifacts.

Outputs include `private_checkpoints/training_state.pt`, `private_checkpoints/hifigan.pt`, `training_history.json`, `evaluation.json` and `dev_reconstructed.wav`. **Never distribute dev source recordings or checkpoints derived from unreviewed voices.** The output reconstructed WAV requires a listener to compare it with the licensed original. A finite GAN loss or the appearance of WAV is not proof of quality.

## Superpowers workflow

Apply the upstream [`obra/superpowers` using-superpowers](https://github.com/obra/superpowers/tree/main/skills/using-superpowers) with brainstorming/design -> writing-plans -> isolated branch -> TDD -> verification -> code review. Read the design and plan under `docs/superpowers/` before changing the feature. Do not merge merely because CI passes.
