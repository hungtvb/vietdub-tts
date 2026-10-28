# Optional Adam synthetic-voice corpus for native VietDub TTS

Source: https://huggingface.co/datasets/Phitran21/Vietnamese-Adam-TTS-50h

This is **synthetic Adam-preset speech**, not recordings of a real human voice. Its data card currently describes 18,478 WAV/text pairs, ~50.55 hours, 48-kHz mono PCM16 and one `finnal_tts.zip` (~17.5 GB), with `data.jsonl` rows `{"id": 1, "text": "..."}` and `wav/<id>.wav`. The card labels the dataset **CC BY 4.0** but explicitly warns that upstream voice-generation and source rights must be checked independently. The listed count and duration are not full-file audited; Hugging Face's Dataset Viewer currently errors. **Do not equate the metadata with verified voice-training authorization.**

## Import (offline, bounded, no ZIP extraction)

Obtain a copy of `finnal_tts.zip` yourself in a private storage area after checking the upstream terms. The importer does not use network APIs, load pretrained weights, extract unsafe ZIP paths, or distribute copyrighted recordings.

```bash
pip install -e '.[dev]'
# Review-only: the vocoder will refuse to train on these rows.
vietdub-adam-vocoder \
  --archive /private/data/finnal_tts.zip \
  --output-dir /private/work/adam-review \
  --max-clips 2000
```

Inspect `import_report.json`, a sample of `audio/*.wav`, and `transcripts_review.jsonl`. This emits **mono 22050-Hz** prepared WAVs, with amplitude/clipping, declared source-format and duration validation, source fingerprint deduplication and provenance. No automatic proof of pronunciation, language coverage, identity, transcript accuracy or license applies.

**If and only if you have personally verified BOTH the corpus license AND upstream Adam voice/generator rights** and confirmed that they permit the intended use, prepare to a *new* approved folder with all three explicit confirmations:

```bash
vietdub-adam-vocoder \
  --archive /private/data/finnal_tts.zip \
  --output-dir /private/work/adam-approved \
  --max-clips 2000 \
  --acknowledge-dataset-rights-reviewed \
  --acknowledge-upstream-voice-rights-reviewed \
  --approve-vocoder-training
```

Flags are **operator assertions only**. This assistant has not independently verified upstream voice-generation rights and has **not** run a real Adam GPU training session. Never set those flags just to bypass the gate.

```bash
python -m vietdub_native.hifigan_train \
  --manifest /private/work/adam-approved/vocoder.jsonl \
  --output-dir /private/runs/adam-vocoder-v1 \
  --steps 50000 --device cuda --checkpoint-every 250 \
  --eval-seconds 5
```

`--steps` is total optimization steps, not additional steps. You may resume after interruption with `--resume` and exactly the same private source files, config, and dataset manifest. Run loss and quality review; a generated WAV by itself does **not** prove that the speech is intelligible.

## What train/dev means here

This dataset has **one synthetic voice preset**. A speaker-disjoint train/dev partition is impossible. The importer instead groups identical Unicode-NFC/casefold/whitespace-normalized utterances to prevent the same sentence from appearing in both splits. It also deduplicates source and output WAV bytes. This measures **unseen-sentence reconstruction** of a *single voice*, not generalization to a held-out speaker or novel text-to-speech inference. There is **no separate genuine human speech evaluation** without another corpus.

The vocoder manifest includes `text` for provenance but the generator trains from mel/audio only; it needs neither forced phoneme alignment nor emotion labels. `transcripts_review.jsonl` is always an **unapproved review queue** (`alignment_verified=false`, `transcript_verified=false`, no `token_durations`, unverified neutral placeholder). Acoustic TTS training is **not enabled** automatically.

## Kaggle / hardware

The ZIP itself is ~17.5 GB. Attaching it as an external Kaggle Input is preferable to copying it into `/kaggle/working`. The importer reads each selected member directly via `zipfile` and materializes *only the capped prepared samples*. Storage still must hold the input ZIP and generated samples/checkpoints; verify space before a full import. See [`notebooks/kaggle_adam_tts_gpu.ipynb`](../notebooks/kaggle_adam_tts_gpu.ipynb). It defaults to non-approved and refuses to run GPU training before explicit review.

## Limits

- Synthetic single-preset speech may transfer teacher pronunciation artifacts, noise, and unnatural prosody. It is an **optional synthetic distillation experiment**, not a replacement for real consented voice recordings.
- The source TTS engine's voice model/asset terms have not been independently resolved here; the dataset card's license label alone is insufficient.
- Dataset-viewer failures are handled by reading a locally supplied archive; they are not evidence that all 18,478 clips are valid.
- Review, checkpoint, report and waveform artifacts remain private; do not automatically publish checkpoint/audio under a voice identity without authorization.
