# FOSD v4: original timestamp TXT + MP3 preprocessing

**Source:** [FPT Open Speech Dataset v4](https://data.mendeley.com/datasets/k9sxg2twv4/4), FPT Corporation, distributed by Duc Chung Tran, DOI:10.17632/k9sxg2twv4.4. The publisher indicates CC BY 4.0 with an additional FPT permission and copyright notice. Preserve both and verify voice-use rights, especially for commercial dubbing. Software cannot verify contributor consent.

The format described in research is:
audio_file_name|transcript|speech_start-speech_end [speech_start-speech_end ...]

Only a **single interval per transcript row** (or a whole-recording transcript) is supported. Multiple intervals with one transcript are quarantined rather than inventing a text-to-time mapping. Accepted times are decimal seconds or [HH:]MM:SS fractions.

Download the authorized original FOSD dataset following the publisher instructions, inspect its license and notice, and extract it *locally*. Audio is never uploaded to GitHub.

## Prepare reviewable segments

    pip install -e '.[dev,api]'
    vietdub-fosd --transcript data/raw/fosd/Set001/transcripts.txt \
      --audio-root data/raw/fosd \
      --output-dir data/staged/fosd \
      --acknowledge-rights --acknowledge-terms

Pass repeated --transcript flags for additional TXT manifests. Filename entries are resolved relative to --audio-root. This command NEVER downloads datasets automatically.

Output files:
- clips/<sha>.wav: 22050 Hz mono PCM16 cuts with original source timestamps.
- metadata_for_review.csv: transcript, clip path, source SHA256 and times. **speaker_id is blank.**
- rejected.jsonl: ambiguous intervals, missing audio, unsafe paths, invalid time, low audio quality, duplicates.
- prepare_report.json: clip statistics and explicit ready_for_tts_training=false.

Review transcripts against audio, assign speaker IDs based on actual verified voice identity and retain source licensing. The filename is NOT a validated speaker ID. Multi-interval rows require human editing before reprocessing.

## Import into the existing QA pipeline

    export VIETDUB_DATA_SALT="$(openssl rand -hex 16)"
    vietdub-datasets import \
      --source fpt_fosd_v4 --format table \
      --metadata data/staged/fosd/metadata_for_review.csv \
      --audio-root data/staged/fosd/clips \
      --output-dir data/prepared/fosd \
      --speaker-salt-env VIETDUB_DATA_SALT \
      --acknowledge-rights --acknowledge-terms

If speaker IDs have not been reviewed, all those clips remain in review_unassigned and cannot be used for training. No token-duration, actor-consent or emotion annotations are fabricated. Run vietdub-audit after importing and independently review the CTC alignment proposals.

## Evidence boundary

The automated tests use locally generated synthetic WAV and MP3 *only to validate infrastructure* (timestamp cuts, mono conversion, safe paths, ambiguous input rejection and importer compatibility). A real FOSD download, real speech checkpoint, and verified listening benchmark have NOT been performed by this change.

## Dataset caution

Unofficial mirrors can carry different licenses; do not conflate the original FPT/Mendeley distribution terms with the license of an unrelated republisher. If using a mirror, inspect its own terms and provenance.