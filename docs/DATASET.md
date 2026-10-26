# Dataset contract & rights

Each training item needs text, actor identity, a fixed emotion label, emotion intensity, original WAV and accurate per-normalized-grapheme mel-frame durations. Tokenizer is versioned in checkpoints. Never reuse checkpoints after changing it. Never invent durations from equal character spacing: alignment must be checked against original speech. Exactly `sum(token_durations) == number of mel frames` for each processed WAV; the loader verifies this.

## Recommended collection and QA

1. Obtain explicit rights and informed voice consent from narrators/actors. Document permitted uses and deletion policy.
2. Record expressive Vietnamese acting across neutral/sad/angry/fearful/crying/whispering/happy/surprised scenes. Balance actor and emotion combinations to prevent confounding speaker ID with emotion.
3. Human-correct all transcripts and emotion labels. Keep train/dev/test clips split by source recording sessions to avoid near-duplicate leakage.
4. Run an independently verified speech-to-grapheme forced alignment workflow producing mel-frame durations using the exact configuration (22,050 Hz / hop 256 / STFT center=True). This repo intentionally does not ship or pretend to solve forced alignment yet.
5. Examine silent clips, noise, clipping, transcript mismatches, speaker leakage, inaccurate alignments and multilingual loanwords before training.
6. Create an emotion-balanced **blind** evaluation: naturalness MOS, emotion classification by listeners, speaker similarity by distinct evaluators, WER/CER via external STT for omissions, time-slot deviation, RTF, clip quality.

### Known blockers

- Vietnamese phoneme/G2P normalization and abbreviations/numerals
- Verified aligner that supports graphemes plus silence/punctuation
- Prosody supervision (F0, energy, emotional arcs) and acting reference conditioning
- Professional high-fidelity adversarial vocoder, room effects and audio QA
- GPU-scale training, dataset licenses and independent listening evaluations
