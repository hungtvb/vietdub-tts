# Native HiFi-GAN-inspired vocoder — experimental training

This is an **independently initialized waveform generator**, not VieNeu, a pretrained checkpoint, or a guaranteed intelligible TTS model. It only reconstructs audio from mel-spectrograms. Text-to-speech will require a separately trained acoustic model.

## Input manifest

Create a **locally verified** `vocoder.jsonl` alongside mono 22050 Hz WAV files. Each row requires `audio_path`, `dataset_source`, `split` (`train`/`dev`), `license`, `source_url`, `attribution`, `rights_verified: true` and `vocoder_training_approved: true`. For source `actor_recordings`, also require `voice_training_consent_verified: true` following consent review.

This is a schema example only. Do **not** set approval fields to true until actual rights were reviewed. The original FOSD probe remains `review_unassigned` and fails this gate intentionally.

```json
{"audio_path":"audio/approved_001.wav","split":"train","dataset_source":"approved_recordings","license":"PER-RECORDING LICENSE REVIEW REQUIRED","source_url":"https://example.org/provenance","attribution":"RECORDING OWNER TO VERIFY","rights_verified":true,"vocoder_training_approved":true}
```

The trainer requires two or more approved train clips and at least one approved held-out dev clip. It does not need transcripts, token durations or emotion labels. It rejects duplicate source hashes and duplicate prepared WAV contents across train/dev, path escapes, non-mono audio or unexpected sample rates. `rights_verified` is an assertion, not proof of legal permission.

## Prepare a larger Vietnamese corpus — Mozilla Common Voice

Official catalog: https://commonvoice.mozilla.org/ (the Vietnamese scripted-speech release is listed as CC0-1.0). Download the release **through the official Mozilla Data Collective**, verify its exact version/license, and unpack it privately so that `validated.tsv` and `clips/*.mp3` are present. Do not bypass login/access terms or upload source voice files to this public GitHub repository.

Vocoder-only training does **not** need transcriptions or forced alignment. The preparer uses `client_id` only to keep speakers out of both train and dev, reserves clips from two distinct train speakers for even tiny bounded datasets, and never writes raw client IDs to its output. It also rejects silent/clipped clips and reports quality rejects. Rows without known `client_id` are skipped rather than silently treated as different people. It records original audio digests and converts bounded clips to mono 22050 Hz WAV. Source files remain untouched.

First produce an **unapproved review-only dataset**:

```bash
vietdub-commonvoice-vocoder \
  --source-root /kaggle/input/common-voice-vi \
  --output-dir /kaggle/working/vietdub-vocoder-review \
  --max-clips 10000
```

Inspect `import_report.json`, source provenance and clips. Without operator approvals, `train_hifigan` refuses this manifest. After separately verifying the **specific release's licensing, contributor terms and permitted voice uses**, prepare a fresh approved directory with both deliberate confirmations:

```bash
vietdub-commonvoice-vocoder \
  --source-root /kaggle/input/common-voice-vi \
  --output-dir /kaggle/working/vietdub-vocoder-approved \
  --max-clips 10000 \
  --acknowledge-rights-reviewed --approve-vocoder-training
```

The two flags reflect **operator assertions**: the importer does not independently prove them. Original audio and generated training checkpoints should be stored privately. Before claiming a new model works, run GPU training for a meaningful number of steps and independently listen to held-out reconstructed audio.

## Kaggle GPU execution

A non-executed, review-gated starter notebook is included at [`notebooks/kaggle_vocoder_gpu.ipynb`](../notebooks/kaggle_vocoder_gpu.ipynb). Attach a *licensed* Common Voice Vietnamese release as Kaggle input and edit `SOURCE_ROOT`. Both `RIGHTS_REVIEWED` and `APPROVE_VOCODER_TRAINING` default to `False`; the notebook intentionally refuses to train until the operator approves the actual source. It clones this feature branch, imports speaker-disjoint clips, and trains on `cuda` with resumable checkpoints. Kaggle does not provide durable artifact persistence by default: save the private `training_state.pt` outside ephemeral working storage to continue training after a session. No Kaggle GPU job has been executed by this change.

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

## Training-quality update: joint period/scale critics (checkpoint v3)

The next experiment uses **both** multi-period and multi-scale waveform
critics. The generator objective includes **differentiable log-mel L1** in
addition to multi-resolution STFT, feature matching and adversarial losses.
These changes constrain phonetic/spectral structure more directly than the
period-only experiment, but are **not evidence of intelligible Vietnamese**.
The scale critics are lightweight, HiFi-GAN-inspired and not a reproduction
of any upstream pretrained architecture or weights.

**Checkpoint compatibility:** This changes discriminator parameters and
optimizer state. The checkpoint schema is now
`vietdub_hifigan_training_v3`. Do **not** `--resume` from v2 checkpoints;
start a **new private output directory** for the v3 architecture. Within a v3
run, identical input/audio/config resumability is still tested. Do not copy
an old `training_state.pt` and assume it has been converted.

The held-out `evaluation.json` now includes `dev_logmel_l1`,
`dev_silence_logmel_l1`, `dev_logmel_l1_ratio_to_silence`, source RMS and
reconstruction clipping fraction. A relative log-mel error smaller than 1
means the generated audio is spectrally closer to the reference than silence
under this **one metric**; it does **not** mean intelligible speech. Before
marking the vocoder acceptable, a reviewer with the appropriate rights must
listen to multiple held-out speech clips and compare with originals.

**Critical remaining bottleneck:** No real licensed Vietnamese corpus or GPU
training has been executed by this ChatGPT run. CI tests use synthetic sine
waveforms; passing them cannot constitute a speech-quality claim.

### GPU log-mel device safety

The differentiable mel reconstruction objective analyzes *generated tensors on
CUDA* as well as CPU reference data. `audio.mel_spectrogram` now caches its
analysis transform by tensor device and hyperparameters, moving filter-bank
buffers to the waveform's device. Previously the cached transform was always
on CPU, which could cause a CUDA device mismatch during loss evaluation.
A CUDA-specific backward test is provided and is **skipped on CPU runners**.
Actual CUDA training remains unverified until a GPU run completes.
