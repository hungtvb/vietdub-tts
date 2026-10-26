# VietDub TTS

**Independent Vietnamese speech synthesis for film dubbing**, with a stable character voice registry, timestamp-aligned WAV segments, explicit emotion-control capability reporting and an offline-testable service layer.

> **Status — Phase 1 MVP.** This is VietDub's own service and API, **not yet a separately trained neural model**. The optional inference adapter uses [VieNeu-TTS v3 Turbo](https://github.com/pnnbao97/VieNeu-TTS) (Apache-2.0). Emotional acting is **not guaranteed**: the v3 `style` argument is deprecated; only supported experimental cues are forwarded. Phase 2+ will add proper emotionally labeled datasets, evaluation and fine-tuning.

## Quick start

Python 3.10+ is required. FFmpeg is recommended for gentle speed-up of overlong dialogue (up to 1.25x). The CPU model may download weights on first use.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -e '.[model]'
vietdub-tts serve --host 127.0.0.1 --port 8000
```

To test without downloading any model:

```bash
pip install -e '.[dev]'
pytest -q
```

Single sentence WAV:

```bash
vietdub-tts speak 'Em không thể tin được!' --voice 'Mai Anh' --emotion sad --output line.wav
```

### Speech endpoint

```bash
curl -sS http://127.0.0.1:8000/v1/speech \\
  -H 'Content-Type: application/json' \\
  -d '{"text":"Đừng bỏ em lại!","voice_id":"Mai Anh","emotion":"crying"}' \\
  --output line.wav
```

The `X-Emotion-Control` response header honestly reports `neutral`, `experimental_cue` or `uncontrolled`. `crying` does *not* magically cause crying on the current model.

### Batch film dubbing endpoint

```bash
export VIETDUB_TTS_VOICES_FILE="$(pwd)/examples/voices.example.json"
vietdub-tts serve
```

```bash
curl -sS http://127.0.0.1:8000/v1/dubbing/segments \\
  -H 'Content-Type: application/json' \\
  -d '{"segments":[{"index":0,"start":0.5,"end":3.5,"speaker_id":"female_lead","text":"Anh đã hứa sẽ không bao giờ bỏ em mà!","emotion":"crying"}]}' \\
  --output segments.zip
```

ZIP contains `seg_0000.wav` and `manifest.json` with timestamps, speaker, generated duration, emotion metadata and 44.1 kHz mono PCM16 output compatible with VietDub's dubbing mix pipeline. A too-long line that needs >1.25x speed-up returns HTTP 422: adjust the translation or split the sentence; the service does not silently clip spoken words. Shorter audio receives trailing silence.

**Character consistency:** Configure `VIETDUB_TTS_VOICES_FILE` with `characters` keyed by the human-reviewed `speaker_id`. Each maps to exactly one preset voice or a local `ref_audio` path. Unknown characters are rejected rather than being quietly assigned the narrator's voice. The API deliberately does not accept arbitrary audio file paths from clients. Use only voice recordings you are authorized to reproduce.

**Security:** Run on loopback by default. The API has no authentication or rate limits yet; do not expose directly to the public internet. For remote setups add a proxy with authentication, TLS, request limits and job scheduling.

See [VietDub integration](docs/integration.md) and [development roadmap](docs/roadmap.md).
