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
