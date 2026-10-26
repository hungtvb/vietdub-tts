# Native expressive TTS plan

## P0 – Own the weights and reproducible training [this PR]

Trainable Vietnamese acoustic and waveform neural nets, speaker/emotion embeddings, alignment-validated manifest, independent checkpoints, minimal CLI/API, CI smoke train. Do **not** depend on VieNeu or other TTS backends. GPU training quality is **not verified**.

## P1 – Real Vietnamese preprocessing + validated alignment

Normalize Vietnamese numbers, dates, names and punctuation. Implement G2P/phonemes. Train monotonic alignment/duration modelling or plug in auditable aligner for data preparation; verify voiced/silent timing. Add speech/music separation for licensed film data and stricter dataset provenance.

## P2 – Acting and emotion prosody

Emotion-balanced recordings. Explicit F0/energy/intensity modeling and dialogue-context encoder with preceding/following lines. Reference-acting embedding trained to preserve speaker identity while controlling emotional expression. Human listening A/B comparisons. Avoid inferring emotions only from subtitles when the original performance is available.

## P3 – Production neural vocoder and acoustic quality

Add adversarial waveform and multi-period discriminators, spectral and perceptual losses, multi-band audio QA, clipping/repetition checks. Run genuine CPU/GPU benchmarks on held-out speakers and styles.

## P4 – Film timing and VietDub adapter

Generate takes at target segment duration with model-level duration control; select by ASR text fidelity, speaker match, subjective emotion scores and acoustic artifacts; if required, request adapted translations instead of crushing speech with time stretching. Preserve overlapping conversations and speaker mapping. Integrate into VietDub via separate PR only after quality is demonstrably acceptable.
