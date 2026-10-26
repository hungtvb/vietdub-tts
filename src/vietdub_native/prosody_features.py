"""Pitch/energy diagnostics from waveforms, without pretrained speech models.

Naive normalized autocorrelation is a QA instrument, not a phonation/style
classifier or a certified F0 tracker. In noisy/creaky voices, half/double
pitch artifacts are expected; high-stakes labeling requires reviewed F0.
"""
from __future__ import annotations

import numpy as np


def extract_prosody(wave: np.ndarray, sr: int = 22050, *,
                    frame_size: int = 2048, hop: int = 256,
                    min_hz: float = 70.0, max_hz: float = 450.0,
                    voiced_threshold: float = .48) -> dict:
    """Get aggregate pitch and frame RMS diagnostics, not emotion inference.

    Uses FFT autocorrelation of overlapping Hann windows, with a minimum RMS
    gate and peak autocorrelation threshold. Returns None for unavailable F0.
    """
    samples = np.asarray(wave, dtype=np.float32).reshape(-1)
    if sr < 8000 or frame_size <= 32 or hop < 1 or not 0 < min_hz < max_hz < sr/2:
        raise ValueError("Invalid pitch analysis configuration")
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("Invalid waveform")
    if samples.size < frame_size:
        return {"pitch_median_hz": None, "pitch_p10_hz": None,
                "pitch_p90_hz": None, "voiced_frame_ratio": 0.0,
                "rms_mean": float(np.sqrt(np.mean(samples.astype('float64')**2))),
                "rms_cv": None, "pitch_method": "fft_autocorr_reference_v1"}
    from numpy.lib.stride_tricks import sliding_window_view
    frames = sliding_window_view(samples, frame_size)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1))
    windowed = frames.astype(np.float64) * np.hanning(frame_size)[None, :]
    # Zero padding ensures linear (not wrapped circular) correlation.
    size = 1 << (2*frame_size-1).bit_length()
    spectra = np.fft.rfft(windowed, n=size, axis=1)
    acf = np.fft.irfft(spectra * np.conj(spectra), n=size, axis=1)
    first = max(1, int(np.floor(sr/max_hz)))
    last = min(frame_size-1, int(np.ceil(sr/min_hz)))
    lag_scores = acf[:, first:last+1] / (acf[:, :1] + 1e-12)
    lag_idx = lag_scores.argmax(axis=1) + first
    strength = lag_scores.max(axis=1)
    voice_mask = (rms >= max(0.002, float(np.median(rms))*0.15)) & (strength >= voiced_threshold)
    f0 = sr / lag_idx[voice_mask]
    active = rms > 0.002
    mean_rms = float(np.mean(rms[active])) if active.any() else 0.0
    cv = (float(np.std(rms[active]) / mean_rms)
          if active.any() and mean_rms > 0 else None)
    return {"pitch_median_hz": round(float(np.median(f0)), 2) if f0.size else None,
            "pitch_p10_hz": round(float(np.percentile(f0, 10)), 2) if f0.size else None,
            "pitch_p90_hz": round(float(np.percentile(f0, 90)), 2) if f0.size else None,
            "voiced_frame_ratio": round(float(voice_mask.mean()), 4),
            "rms_mean": round(mean_rms, 6),
            "rms_cv": round(cv, 4) if cv is not None else None,
            "pitch_method": "fft_autocorr_reference_v1"}
