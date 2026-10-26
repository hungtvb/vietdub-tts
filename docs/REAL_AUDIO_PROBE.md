# FOSD real audio research probe

Uses the Hugging Face dataset viewer and an **unofficial mirror** `awghuku/fosd` to inspect a *bounded* number of genuine recorded Vietnamese waveforms from FPT Open Speech Dataset. Original publisher/attribution/permission notice: https://data.mendeley.com/datasets/k9sxg2twv4/4 . The original includes an FPT permission notice as well as CC BY 4.0; preserve attribution, copyright and modification notices. Mirror provenance and voice rights should be manually confirmed before subsequent dataset use.

## Run the probe

    pip install numpy soundfile soxr
    PYTHONPATH=src python -m vietdub_native.real_audio_probe \
      --output data/real-audio-probe --limit 6 --min-ok 2 --acknowledge-rights

On GitHub, push changes to the probe paths on main to trigger **Real Vietnamese Audio Probe** action. It downloads at most six audio clips with maximum 3 MiB each (per response), uses ffmpeg to decode to WAV 22,050 Hz, checks length, peak, RMS, clipping and preliminary F0/voicing/energy, and writes `report.json`. The FFT-autocorrelation pitch estimator is an uncalibrated diagnostic and may contain octave errors; it is not a trained pitch/emotion recognizer. GitHub artifact uploads **only** this anonymized summary, not raw recordings or transcriptions. The files remain in the Actions runner and are not committed.

## Strict limitations

- Success proves actual decoded audio was inspected in Actions, not that the model learned to pronounce Vietnamese or express emotions. Prosody features are observational diagnostics only.
- Public mirror and publisher license must be reviewed; no voice identity is assumed from filename or clip.
- Manifest has `speaker_id=unassigned`, `transcript_verified=false`, `alignment_verified=false`, `emotion_status=unverified_placeholder`, and `split=review_unassigned`. It MUST NOT be used to train acoustic/vocoder models without human approval.
- Audio samples may contain noise, clipping, music, transcript errors, different speakers, or multiple emotions.
- The repository never downloads a pretrained TTS checkpoint and no trained checkpoint is released by this probe.
- The HF Dataset Viewer API, dataset availability and cached-audio URLs can change. If external access is unavailable the workflow should fail rather than pretending the samples are real.
