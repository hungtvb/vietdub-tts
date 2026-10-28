# Verify the real Adam 50h archive without downloading 17.5 GB

Source: https://huggingface.co/datasets/Phitran21/Vietnamese-Adam-TTS-50h

The dataset card reports 18,478 generated single-preset Adam Vietnamese speech clips (~50.55 h), distributed in `finnal_tts.zip` (~17.5 GB). The Dataset Viewer currently errors, and its metadata counts have not been fully verified by decoding all recordings.

`vietdub_native.adam_remote_probe` inspects the **remote ZIP central directory** and at most three WAVs over HTTP Range GET. It requires exact `206` and `Content-Range`, caps transfer to 16 MiB, validates 48 kHz/mono/PCM16/finite signal/duration/RMS, and outputs *aggregate metadata only*. No ZIP, transcript, source waveform or checkpoint is saved, uploaded, or published.

Run after installing `numpy soundfile soxr`:

```bash
PYTHONPATH=src python -m vietdub_native.adam_remote_probe \
  --output data/adam-source-qa.json --limit 3
```

GitHub workflow `Adam 50h Remote Source Probe` tests the HTTP client locally, then attempts the real remote probe. **Expected fail-closed behavior:** if the Hugging Face/Xet CDN does not honor Range requests, the workflow fails and logs the reason; it does **not** silently download the complete 17.5 GB ZIP. CDN redirects or cached download endpoints can change.

Important: the source WAVs are synthetic Adam voice data. This probe **does not** train any model, confirm TTS intelligibility, or establish upstream voice/generator rights. The dataset license label does not resolve those rights; the existing training gate remains OFF. Do not use successful source QA to justify model release.

## 2026-10-28 real-source finding and importer correction

The first live HTTP Range attempt reached the real `data.jsonl` (so partial ZIP
access works), but the strict importer aborted at source metadata line **1211**:
`text must be a nonempty Vietnamese string (<=1000 chars)`. The source contains
at least one transcript outside our local validation policy. Do not display,
truncate, or silently accept it. The corrected importer now **quarantines**
blank/oversized text rows, adds `rejected_invalid_transcript` to its report,
and proceeds with valid rows. The remote QA reports
`metadata_invalid_transcripts`; malformed JSON or repeated/missing IDs remain
fatal. This is an actual dataset quality finding, not a synthetic fixture.

## Remote subset import: no 17.5 GB attachment needed

In addition to metadata inspection, the exact same seekable reader can now be
passed directly into the existing `prepare_adam_tts` importer. Only *selected*
WAV members are read and downsampled. For preview-only without human rights
review (this does NOT train):

```bash
PYTHONPATH=src python -m vietdub_native.adam_remote_probe \
  --output data/adam-import-qa.json \
  --prepare-dir data/adam-review --prepare-clips 3
```

For a private GPU experiment, `prepare_remote_adam_tts()` supports up to 500
selected clips with an explicit transfer cap; defaults stay review-only. The
Kaggle notebook now defaults to this *remote* mode (250 clips, 2 GiB cap), but
**will not run training** without an operator reviewing and attesting both
corpus rights and upstream synthetic voice/model rights. Use the local ZIP
importer for larger-scale training once rights are resolved. HTTP Range
availability is not guaranteed on Kaggle; if unavailable, use a mounted ZIP.

The remote preview output directory includes prepared source WAVs **privately**
while the CI runner is alive. GitHub uploads **only JSON QA reports**, never
these recordings or model checkpoints.
