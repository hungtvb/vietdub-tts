# Optional Adam-50h synthetic-corpus intake — design

## Intent

Allow the native VietDub vocoder to **optionally** consume Phitran21's single-preset synthetic Adam data without falsely claiming those recordings are human voices or granting automatic training rights. Users need a workable importer for a single 17.5-GB ZIP, not an unavailable online dataset viewer.

## Constraints and non-goals

- Keep current HiFi-GAN training API, explicit operator rights gate, checkpoint format, current Common Voice importer, and **native weights**.
- No automatic download of ~17.5 GB, no ZIP extraction all-at-once, no new model dependency, no auto-approving upstream voice ownership.
- This is **not** native Vietnamese TTS inference, emotion modeling, a verified transcription-alignment tool, or a finished audio demo.
- A lone synthetic preset cannot yield speaker-disjoint evaluation; use normalized-utterance group-disjoint split and document this limitation.

## Approach

Choose small dedicated offline ZIP importer over changing Common Voice's multi-speaker TSV processor or relying on HF Dataset Viewer. It recognizes one `data.jsonl` under an optional directory prefix and `wav/<numeric-id>.wav`. Validate safe member names, exact 48-kHz mono PCM16 source format, bounds/quality, duplicate WAVs and duplicate normalized text groups. Convert selected WAVs to 22,050 Hz and record provenance. Optionally output review-only acoustic JSONL without manufactured timing. Unapproved by default, require corpus and **upstream voice rights** review before approving vocoder training.

## Acceptance

- Tests demonstrate RED then GREEN on missing importer and review-only manifest behavior.
- Invalid WAV and unsafe archive entries must be rejected; deterministic text-group split prevents repeated transcripts across train/dev; strict voice-rights gate cannot be bypassed by corpus license alone.
- Approved synthetic fixture can feed one native HiFi-GAN optimizer step, but only a full-length held-out, rights-cleared *real* reconstruction listening check may certify sound quality.
- No claim of actual Adam audio training until archive obtained and GPU run + human listening documented.
