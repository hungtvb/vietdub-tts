# VietDub native CTC alignment

The alignment model is developed for VietDub in PyTorch and starts from random
weights. It requires no VieNeu, Whisper, pretrained TTS or third-party speech API.
Training requires clean consented voice recordings and human-corrected transcripts.

## Training

A JSONL transcript row needs:
- text: the spoken sentence
- audio_path: path relative to manifest
- speaker_id: character voice identifier
- emotion: neutral/happy/sad/angry/fearful/crying/whispering/surprised
- emotion_intensity: optional 0..1 value
- tokenizer: optional, grapheme or vi_phonetic_v1

No per-token durations are needed to train the CTC aligner.

    vietdub-native train-aligner --manifest data/transcripts.jsonl \
      --checkpoint runs/aligner.pt --tokenizer vi_phonetic_v1 \
      --device cuda --epochs 30

Example epoch count is illustrative; no real GPU/actor benchmark has been performed.

## Create proposals

    vietdub-native align-manifest --manifest data/transcripts.jsonl \
      --checkpoint runs/aligner.pt --output data/review/proposed.jsonl

The output has token_durations, tokenizer, alignment_method, alignment_confidence,
alignment_status=needs_human_review, alignment_verified=false.

Viterbi CTC decoding uses a monotonic path, assigning blank frames to adjacent
tokens. Every frame is accounted for. The score is an uncalibrated model
posterior, NOT a verified boundary confidence.

The input manifest is left unchanged, and audio paths are rebased if output
is in another folder. Checkpoint metadata locks tokenizer vocabulary.

## Mandatory human review

The generated manifest is NOT training-approved. A reviewer checks the
waveform/spectrogram and spoken text, corrects timing and annotations, and then
marks alignment_verified=true. Acoustic training rejects explicitly false
alignment_verified flags, including in grapheme mode; the phonetic mode
requires true and a nonempty alignment_method.

Random-weight/synthetic smoke testing cannot validate phonetic accuracy,
speaker identity, or emotions. Full model training needs real data, held-out
manually labeled ground truth and boundary-error measurement. Do not force
alignment of overlapped dialogue, music, missing words or noisy clips without
careful separate processing.
