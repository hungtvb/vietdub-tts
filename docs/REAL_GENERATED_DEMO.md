# Genuine-data native TTS generation experiment (research only)

This is the first **attempt to synthesize a new WAV using VietDub's own trainable
PyTorch acoustic and vocoder models, after training them on real Vietnamese
source recordings**. It is separate from production acoustic training and
**does not** mark CTC alignments, transcripts, speakers or emotion labels as
human-verified. The result is a pipeline demonstration, not an intelligible,
human-acceptable voice. Training uses random initial parameters: no VieNeu,
F5-TTS, Edge-TTS, pretrained aligner or pretrained vocoder is imported.

## Data and rights

The bounded six-clip probe uses the third-party `awghuku/fosd` mirror of FPT
Open Speech Dataset. Original publisher and license notice:
https://data.mendeley.com/datasets/k9sxg2twv4/4 . Permission, attribution
and performer voice rights require independent review. This research workflow
is opt-in and does not produce a public checkpoint/voice clone.

## Run in a networked environment

1. Download public FOSD research clips to an isolated workspace with
   `python -m vietdub_native.real_audio_probe --output data/real-audio-probe --limit 6 --min-ok 6 --acknowledge-rights`.
2. Run `python -m vietdub_native.research_training --manifest data/real-audio-probe/transcripts.jsonl --output-dir data/native-real-demo --acknowledge-source-rights-reviewed`.
3. Evaluate `data/native-real-demo/vietdub-tts-demo.wav` and its diagnostic JSON.

The optional GitHub Actions `Real Native TTS Demo` workflow performs this
bounded, non-production experiment and publishes only:
- `vietdub-tts-demo.wav` (new model output, not original audio)
- `research_report.json` (anonymized technical evidence)
- `NOTICE.txt` (publisher / rights notice)

The original recordings and model checkpoints remain on the temporary runner;
no raw audio, transcript, speaker identity or trained weights are artifacts.
Artifact retention is limited. GitHub Actions time and network limits apply.

## Strict interpretation

- The weak CTC duration vectors are generated only for this research command;
  they are not serialized with `alignment_verified=true` and cannot be used for
  the production acoustic training path.
- A **WAV file exists** does not imply the file contains understandable speech.
  The current native vocoder, speaker embedding and duration model are too small
  and undertrained to claim usable film dubbing.
- The six FOSD examples contain unverified speakers and transcripts, so one
  learned embedding represents **mixed/unknown** audio; it is not a speaker clone.
- Next validation requires a legally approved, curated multi-hour speaker
  corpus, ground-truth timing review, independent transcription accuracy and
  blind subjective listening tests.

## v2 waveform-quality gate

A generated WAV can be valid PCM yet carry only a DC bias or imperceptible
variations. The research report now measures *AC RMS*, DC/RMS ratio and
zero-crossing rate. The workflow fails its quality gate unless
`acoustic_signal_present` is true. Passing this gate shows nontrivial waveform
variation, **not intelligible speech**. This was added after the first 0.406s
v1 sample measured `DC/RMS=0.9983` and zero crossings of 0. Training settings
for the second experiment increase CTC/acoustic/vocoder steps to 60/120/100,
respectively. No speaker/emotion approval flag is changed.
