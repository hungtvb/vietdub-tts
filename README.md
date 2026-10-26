# VietDub Native Expressive TTS

**A NEW, trainable, local Vietnamese film-dubbing synthesizer. No VieNeu, Edge-TTS, F5-TTS, external TTS inference, or pretrained TTS checkpoint is used by the model.** This is an early research prototype, **not yet a natural-sounding speech model**.

## What is implemented

- Vietnamese NFC grapheme tokenizer with tone marks (v0 baseline; Vietnamese phoneme/G2P system still needed).
- Trainable **emotion- and speaker-conditioned acoustic model**: grapheme embeddings, convolutional text encoder, trainable duration predictor, deterministic length regulator, mel decoder.
- Native **trainable mel-to-wave convolutional vocoder**, trained from random weights with multi-resolution spectral losses.
- Human-/tool-verified grapheme-to-mel-frame alignment **required**. Mismatched duration totals are rejected; no fabricated proportional alignment.
- Emotion labels: neutral, happy, sad, angry, fearful, crying, whispering, surprised. These are **training labels**, not proof the currently untrained model can portray those emotions.
- Offline dataset validation, acoustic and vocoder training, checkpoint versioning, 44.1 kHz WAV export for VietDub, CLI, optional self-hosted FastAPI service.
- All checkpoint files are created by this repository's training steps; no model download or pretrained weights.

## Limitations (must read)

**This is the skeleton and CPU smoke-train of a new model, not a commercial-quality expressive TTS.** Good results require legally obtained recordings, correct transcripts and durations, speakers performing multiple emotions, substantial training, objective listening tests, better G2P and adversarial neural vocoder improvements. Emotion labels only introduce trainable conditioning; they do not magically generate crying or anger. Audio sample rate during training is 22,050 Hz; 44,100 Hz output is resampled for compatibility and does not increase true bandwidth. The v0 training design uses supervised token durations, with no built-in forced aligner. A file with invented token durations is not real alignment and should not be used to judge speech quality.

## Installation

Python >=3.10. For CPU, install PyTorch from its official CPU wheel channel if necessary.

```bash
pip install -e '.[dev,api]'
pytest -q
```

## Real speech data intake

Do not train on synthetic sine-wave test fixtures. VietDub now provides an **offline, rights-attested corpus importer** with source provenance, audio QA, speaker pseudonymization and review gates. It does **not** auto-download external voices.

```bash
vietdub-datasets sources
# See docs/REAL_DATA.md for real dataset import examples.
```

Supported source metadata: Mozilla Common Voice Vietnamese v27, FPT Open Speech Dataset v4, Google FLEURS Vietnamese, or separately consented actor recordings. Actual source terms and permission to train/distribute must be checked manually. Imports are unaligned, not emotion-verified.

See [real data sourcing and import instructions](docs/REAL_DATA.md).

## Training workflow

A dataset is JSON Lines, each row with text, `audio_path` (relative to the manifest), `speaker_id`, `emotion`, `emotion_intensity` (0..1), and `token_durations` (nonnegative **mel frame counts** for every character after NFC normalization, including spaces). Alignments must be independently produced and verified with `sample_rate=22050`, `hop_length=256`, `n_fft=1024`, mel `center=True`.

```bash
vietdub-native validate-manifest data/train.jsonl
vietdub-native train-acoustic --manifest data/train.jsonl --output-dir runs/film_v1 --epochs 50 --device cuda
vietdub-native train-vocoder --manifest data/train.jsonl --output-dir runs/film_v1 --epochs 100 --device cuda
vietdub-native synthesize \
  --acoustic runs/film_v1/acoustic.pt \
  --vocoder runs/film_v1/vocoder.pt \
  --speaker female_lead --emotion sad --intensity 0.8 \
  --text 'Anh đừng rời xa em!' --output dub.wav
```

Training epoch counts above are illustrative, **not tuned recommendations** or evidence of quality. The supplied `examples/manifest.example.jsonl` illustrates schema only; its audio is not included and durations are placeholders. Never train on it.

Serve locally with environment variables `VIETDUB_ACOUSTIC_CHECKPOINT` and `VIETDUB_VOCODER_CHECKPOINT` then `uvicorn vietdub_native.api:app --host 127.0.0.1 --port 8020`. Endpoint `POST /v1/speech` accepts `text`, `speaker_id`, `emotion`, `emotion_intensity`, and `duration_scale` (0.8..1.2) and returns a WAV. Unknown speakers are rejected rather than silently substituting a voice.

## VietDub integration

`POST /v1/speech` returns a 44,100 Hz mono PCM16 WAV, compatible with VietDub's existing duration-fit stage. A separate VietDub repository PR should add a configurable `native-tts` provider after the native model meets audible-quality acceptance tests. Preserve speaker mappings and the original timestamps; no changes to the parent repository are included here.

## Model provenance

Architecture and training loops were implemented specifically for VietDub. Design principles (non-autoregressive text-to-mel, duration regulation, mel-to-wave upsampling) are common published neural speech ideas, not a claim of a newly discovered academic architecture. Training starts from random weights. Use only speaker audio with appropriate rights/consent and log data provenance.

See [research roadmap](docs/ROADMAP.md) and [dataset spec](docs/DATASET.md).
