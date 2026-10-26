# Real data training readiness (v1)

This gate is intentionally conservative. VietDub's TTS and CTC aligner are
original trainable networks, but **no natural-sounding Vietnamese speech model
can be claimed from synthetic tests**. Dataset metadata cannot prove the truth
of a transcription, performer consent, emotional performance, or legal rights.

## Obtain a licensed dataset

- **FPT Open Speech Dataset (FOSD v4)**: source DOI
  https://data.mendeley.com/datasets/k9sxg2twv4/4 ; the publisher lists 25,921
  utterances (~30h) with audio/transcripts and a FPT permission notice as well
  as CC BY 4.0. Retain copyright, attribution, notices, and modification log.
- **Mozilla Common Voice Vietnamese**: obtain the current permitted version
  via https://datacollective.mozillafoundation.org/ after reviewing terms.
- **Google FLEURS vi_vn**: https://huggingface.co/datasets/google/fleurs ;
  keep the official evaluation subset held out.
- **Consented voice actors**: maintain signed voice-training and usage consent,
  plus deletion/retention rules outside this public repository.

The project importer accepts local, authorized dataset copies. It does **not**
fetch, scrape or redistribute external recordings. The model itself never
loads VieNeu or another TTS model.

## 1. Import recordings

Follow `docs/REAL_DATA.md`, e.g. importing licensed FOSD clips that have first
been paired with human-reviewed transcript CSV rows:

```bash
export VIETDUB_DATA_SALT="$(openssl rand -hex 16)"
vietdub-datasets import \
  --source fpt_fosd_v4 --format table \
  --metadata data/raw/fosd/reviewed.csv \
  --audio-root data/raw/fosd --output-dir data/prepared/fosd \
  --speaker-salt-env VIETDUB_DATA_SALT \
  --acknowledge-rights --acknowledge-terms
```

FOSD's original TXT timing file is **not** automatically parsed by this
CSV importer. Never invent speaker IDs or assume all utterances are neutral.

## 2. Inspect the data and review annotations

The importer writes `transcripts.jsonl` with pending alignment and **unverified
neutral placeholder** emotions. A qualified reviewer must compare every
transcript with the recording, review licensing, source/speaker/session identity
and then annotate `transcript_verified: true` for confirmed lines.

```bash
vietdub-audit data/prepared/fosd/transcripts.jsonl \
  --stage aligner --output data/reports/aligner.json --strict
```

The audit reports why clips are blocked. `--strict` returns exit code 2 when
any train-split row is blocked. Speaker identity is counted but not written to
its report. For `train-aligner` on provenance-tagged corpora, the same gate is
enforced at runtime: an unreviewed train split is refused.

## 3. Align and approve boundaries

```bash
vietdub-native train-aligner --manifest data/approved/transcripts.jsonl \
  --checkpoint runs/aligner.pt --tokenizer grapheme \
  --device cuda --epochs 30
vietdub-native align-manifest \
  --manifest data/approved/transcripts.jsonl \
  --checkpoint runs/aligner.pt \
  --output data/review/proposed.jsonl
```

These commands require the data itself on your machine and sufficient compute.
The CTC proposal writes `alignment_verified: false` and does not certify
phoneme boundaries. Human reviewers must correct boundaries before setting
`alignment_verified: true` and documenting `alignment_method`.

## 4. Verify emotion and voice rights before acoustic training

The `acoustic` stage requires a reviewed transcript, verified alignments,
verified emotion label, stable nonempty speaker ID and source/license metadata.
For actor recordings, include `voice_training_consent_verified: true` only
once the owner has confirmed signed permission. `expressive` additionally
requires multiple verified emotional performances per speaker in the audited
corpus; this is a **minimum structural criterion**, not evidence of quality.

```bash
vietdub-audit data/approved/aligned.jsonl --stage acoustic --strict
vietdub-audit data/approved/aligned.jsonl --stage expressive --strict
```

Both native `train-acoustic` and `train-vocoder` refuse provenance-tagged
train splits that fail acoustic readiness. Legacy synthetic fixtures without
`dataset_source` remain usable for infrastructure tests; they are NEVER
production evidence.

## Known limitations

- The audit checks whether humans **asserted** transcription/emotion/voice
  verification; software does not independently validate those assertions.
- Sample rate and frame counts are checked but not true boundary accuracy.
- Speaker IDs are stable only with the same private salt. Reusing an actor in
  another corpus with a different hashed ID can conceal train/test leakage.
- Cross-split fingerprint checks use the source audio SHA where provided, or
  prepared WAV bytes otherwise; perceptually duplicate re-encodings can evade
  this check.
- There has been **no confirmed real-data download or GPU-scale training** in
  the assistant's execution environment. Obtain licensed assets and run the
  above commands in a networked, GPU-capable environment; record hashes,
  license/consent documentation and independent listening evaluations.
