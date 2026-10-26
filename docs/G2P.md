# Vietnamese text frontend v1 (research)

The new opt-in vi_phonetic_v1 frontend generates onset, vowel-quality/tone,
coda and punctuation units instead of one token per Unicode character. It
expands ordinary Vietnamese numbers, valid dates, times and percentages.

**This is not a verified IPA transcription or a trained grapheme-to-phoneme
model.** Vietnamese regional pronunciation, dialects, ambiguous words,
abbreviations, English proper nouns, dates used as identifiers, and disfluency
require contextual language processing and human review. Unsupported
characters are rejected rather than silently converted to unknown tokens.

## What changes

The original grapheme checkpoint vocabulary is untouched. The model and
checkpoint now store the selected tokenizer mode and the exact vocabulary.
A vi_phonetic_v1 acoustic checkpoint cannot reuse grapheme durations or
be loaded as a grapheme model. The vocoder's audio config must match.

Use --tokenizer vi_phonetic_v1 consistently for **both** acoustic and vocoder
training. A phonetic manifest item must explicitly contain:

- tokenizer: "vi_phonetic_v1"
- token_durations: one mel-frame count per token from
  vietdub_native.symbols.tokenize(text, "vi_phonetic_v1")
- alignment_method: nonempty description of external/human alignment
- alignment_verified: true; human-verified transcript/intervals

The loader verifies duration vector count and total mel frames (22,050 Hz,
256-hop, STFT centered) but **cannot prove the durations are correct**.
Forcing even/proportional token durations is allowed ONLY in unit-test synthetic
fixtures and must never be used as an actual training alignment.

## Practical workflow

1. Record licensed, clean actor speech; verify transcripts and timing.
2. Normalize text using normalize_for_speech, then inspect context-sensitive
   numeral and loanword expansions.
3. Tokenize with vi_phonetic_v1; obtain the true intervals from an
   independently validated aligner or human annotation.
4. Convert boundaries to mel frames, review silence/onsets, ensure frame
   totals equal the recording's computed mel frames, then create the manifest.
5. Run: vietdub-native validate-manifest dataset.jsonl --tokenizer vi_phonetic_v1
6. Train both components with --tokenizer vi_phonetic_v1, each from random
   initialization; evaluate intelligibility before expressive acting.

This release adds a frontend **and a stricter dataset contract**. It does NOT
implement a forced alignment model. An evidence-backed aligner, regional
lexicon and emotion-prosody training are subsequent research milestones.
