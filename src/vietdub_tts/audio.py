"""Normalize generated audio to VietDub's mono PCM16/44.1kHz segment contract."""
from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import soundfile as sf
import soxr

from .engine import SynthesizedAudio

OUTPUT_RATE = 44100
MAX_TEMPO = 1.25


class DurationFitError(ValueError):
    """Generated speech cannot fit the requested timeline without distortion."""


def _pcm_wav(samples: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def render_wav(audio: SynthesizedAudio, target_duration: float | None = None
               ) -> tuple[bytes, float]:
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.ndim != 1 or not samples.size or audio.sample_rate <= 0:
        raise ValueError("engine must output nonempty mono PCM samples")
    if not np.isfinite(samples).all():
        raise ValueError("engine produced NaN or infinite audio")
    if samples.size / audio.sample_rate > 90:
        raise ValueError("generated speech exceeds 90-second segment limit")
    if audio.sample_rate != OUTPUT_RATE:
        samples = soxr.resample(samples, audio.sample_rate, OUTPUT_RATE)
    if target_duration is not None:
        target_n = round(target_duration * OUTPUT_RATE)
        if target_n < 1:
            raise DurationFitError("target duration too short")
        ratio = len(samples) / target_n
        if ratio > MAX_TEMPO:
            raise DurationFitError(f"speech exceeds slot by {ratio:.2f}x; "
                                   "shorten translation or split dialogue before retry")
        if ratio > 1.005:
            if shutil.which("ffmpeg") is None:
                raise DurationFitError("ffmpeg is required to fit longer speech into slot")
            cmd = ["ffmpeg", "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0",
                   "-af", f"atempo={ratio:.6f}", "-ar", str(OUTPUT_RATE), "-ac", "1",
                   "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"]
            try:
                run = subprocess.run(cmd, input=_pcm_wav(samples, OUTPUT_RATE),
                                     capture_output=True, timeout=45, check=False)
            except subprocess.TimeoutExpired as exc:
                raise DurationFitError("ffmpeg tempo conversion timed out") from exc
            if run.returncode != 0:
                raise DurationFitError(f"ffmpeg failed: {run.stderr.decode(errors='replace')[:250]}")
            adjusted, sr = sf.read(io.BytesIO(run.stdout), dtype="float32")
            if sr != OUTPUT_RATE or adjusted.ndim != 1:
                raise DurationFitError("ffmpeg returned invalid output")
            samples = adjusted
        # Rounding and encoder delay can produce a few extra samples.
        # Do not truncate substantial voiced audio: reject >20ms overflow.
        if len(samples) > target_n + round(0.020 * OUTPUT_RATE):
            raise DurationFitError("fitted speech still exceeds slot by >20ms")
        samples = samples[:target_n]
        if len(samples) < target_n:
            samples = np.pad(samples, (0, target_n - len(samples)))
    samples = np.clip(samples, -1.0, 1.0)
    return _pcm_wav(samples, OUTPUT_RATE), len(samples) / OUTPUT_RATE
