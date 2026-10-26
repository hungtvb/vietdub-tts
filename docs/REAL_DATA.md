# Real Vietnamese speech data: sourcing, rights and preparation

Verified against publisher pages on **2026-10-26**. This is a **source review**, not a legal opinion. Respect all current source terms, contributor voice rights and privacy law when downloading, training or publishing weights.

| Dataset | Publisher page | License / restrictions | Proposed purpose |
|---|---|---|---|
| Mozilla Common Voice Scripted Speech **v27.0 (vi)** | https://datacollective.mozillafoundation.org/ | CC0-1.0. Dataset download terms may prohibit rehosting / re-identification. Review the exact version's terms. | Diverse read-speech pronunciations and speakers. |
| FPT Open Speech Dataset **v4** | https://data.mendeley.com/datasets/k9sxg2twv4/4 | Mendeley page displays CC BY 4.0 and reproduces an FPT permission notice requiring copyright and permission notice retention. Do not ignore either. | Approx. 30 hours / 25,921 transcribed clips; filtered speech baseline. |
| Google **FLEURS**, Vietnamese (`vi_vn`) | https://huggingface.co/datasets/google/fleurs | CC BY 4.0; credit dataset/paper. | Preferred held-out independent evaluation; preserve the original test/dev split. |
| Commissioned Vietnamese actors | Local signed recording contract | Specific consent for TTS training, permitted output use, retention/deletion, speaker identity and cloning. | Essential expressive acting across emotion labels and repeated character identities. |

The default examples deliberately omit VIVOS (noncommercial share-alike) and uncertain-rights web video/audio sets. **Do not collect dialogue from films/TV/social video automatically** or treat publicly viewable audio as training-licensed.

## Install

```bash
pip install -e '.[dev,api]'
vietdub-datasets sources
```

Install `ffmpeg` when importing MP3 clips if your `libsndfile` cannot decode them. Download the licensed dataset yourself through the publisher's official UI/API after accepting its terms. Store data outside Git and avoid uploading raw voice recordings into a public repository. The project's `.gitignore` ignores `data/`, WAV, and model checkpoints. Keep all prepared artifacts in `data/`.

## Import Common Voice v27 (offline from an authorized download)

A Mozilla archive typically includes TSV and a `clips/` directory. Choose `validated.tsv` if available and inspect its field names; on some versions the transcript column may be `text` rather than `sentence`.

```bash
export VIETDUB_DATA_SALT="$(openssl rand -hex 16)"
vietdub-datasets import \
  --source mozilla_common_voice_vi_27 \
  --format common_voice \
  --metadata data/raw/cv-vi/validated.tsv \
  --audio-root data/raw/cv-vi/clips \
  --output-dir data/prepared/cv-vi \
  --speaker-salt-env VIETDUB_DATA_SALT \
  --acknowledge-rights --acknowledge-terms
```

If necessary add `--text-column text`. `client_id` is **pseudonymized again** using a private salt so the original speaker identifier is not copied to the output. Store the salt securely if you need stable identities across import runs. Do not attempt re-identification. Train/dev/test assignment groups clips by salted speaker ID to prevent within-import speaker leakage; very small data can have skewed splits. Explicit `--split train|dev|test` preserves a publisher split, but verify that original publisher's partitions have no speaker overlap.

## Import FPT FOSD v4 (from pre-segmented local WAV/MP3)

FOSD's published package includes MP3 and custom TXT transcripts / timestamps. **The generic CSV converter is not a parser for those original TXT files**. First inspect the files and create a reviewed CSV where each row points to one correct **already clipped** speech recording:

```csv
# field headers shown without the comment marker
audio_path,text,speaker_id
clips/0001.mp3,Xin chào,reader_A
```

`audio_path` must be relative to `--audio-root`. **Do not invent speaker identities:** leave `speaker_id` empty if it is unknown; that clip will be tagged `unassigned` and put in a review-only split. Source speaker values are pseudonymized, not published verbatim.

```bash
vietdub-datasets import \
  --source fpt_fosd_v4 --format table \
  --metadata data/raw/fosd/reviewed.csv \
  --audio-root data/raw/fosd --output-dir data/prepared/fosd \
  --speaker-salt-env VIETDUB_DATA_SALT \
  --acknowledge-rights --acknowledge-terms
```

The FPT license and attribution/notice must be preserved for any derivative redistribution where applicable. Review human-voice usage in the intended deployment before training and commercial use.

## Import an actor corpus

The same generic CSV adapter accepts clean actor recordings **only after consent and data-processing terms have been documented outside the public repository**. Emotion labels are never inferred by this importer. A real expressive dataset needs human-reviewed emotions and multiple emotional performances per speaker. The default `neutral` in the import manifest is marked `emotion_status=unverified_placeholder` and **is not ground truth**.

## Output and gates

The importer writes:

- `audio/*.wav`: mono 22,050 Hz PCM16, under the chosen private output directory.
- `transcripts.jsonl`: local paths, transcript, pseudonymous speaker ID, source/license/attribution, audio SHA256, quality numbers and review status.
- `rejected.jsonl`: every rejected input row with a reason (invalid path, duplicate, silence, unsupported text, invalid duration, clipping, etc.).
- `intake_report.json`: accepted/rejected counts, durations, speaker count, review status. The tool cannot prove licenses or signed consents; the operator's flags are acknowledgements only.

This is ready as an **unaligned transcript** for `vietdub-native train-aligner` after reviewing transcripts. It is **not** ready for acoustic training or emotion conditioning:

1. Manually review every transcript and source file, especially accented speech, noise, source identity and ambiguous pronunciations.
2. Train/evaluate the CTC aligner using *real audio*; proposals are not verified timings.
3. Review and correct token timing against waveforms/spectrograms. Set `alignment_verified=true` only after actual human confirmation.
4. Label emotion and speaker identity from the recording/actor records; avoid assuming all read speech is neutral.
5. Split by *real speaker or recording session* for held-out evaluation, and retain source split provenance. Do not include FLEURS' evaluation clips in training.

## Quality limitations

Current automated gates enforce 0.4–20s clip duration, finite amplitude, minimal RMS, and a clipping threshold. They **cannot** prove clean speech, word accuracy, accent, actor identity, consent, absence of copyrighted material, or emotional authenticity. Quality scores are diagnostic only; no dataset is automatically approved for training or model release.

If you do not already have these datasets downloaded, this repository neither fetches them nor claims to have trained on them.
