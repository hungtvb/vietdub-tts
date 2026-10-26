"""Research-only, weakly aligned proof-of-generation using genuine recordings.

This path is deliberately NOT the approved acoustic training path. It does not
alter any data verification flag, infer emotional ground truth or speaker IDs,
and produces no public checkpoint. Run only on a corpus whose training rights
have been independently reviewed; it is not for producing a voice clone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .audio import mel_spectrogram, read_audio, save_wav
from .checkpoint import save_acoustic, save_vocoder
from .config import Config, EMOTIONS
from .model import AcousticModel, acoustic_loss
from .symbols import encode, unknown_characters
from .vocoder import NeuralVocoder, spectral_loss

RESEARCH_VOICE = "research_unverified_mixed_speakers"
EXPECTED_DATASET = "fpt_fosd_v4"
EXPECTED_MIRROR = "awghuku/fosd"


class ResearchError(ValueError):
    pass


def load_research_samples(manifest: str | Path, cfg: Config, *, limit: int = 6) -> list[dict]:
    """Read unreviewed FOSD waveforms; never turn guessed labels into approved labels."""
    path = Path(manifest).resolve()
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not 1 <= limit <= 12:
        raise ResearchError("Research experiment is limited to 1..12 source clips")
    selected = []
    for row in rows[:limit]:
        if (row.get("dataset_source") != EXPECTED_DATASET
                or row.get("dataset_mirror") != EXPECTED_MIRROR
                or row.get("training_approved") is True
                or row.get("transcript_verified") is not False
                or row.get("alignment_verified") is not False
                or row.get("speaker_id") != "unassigned"
                or row.get("emotion_status") != "unverified_placeholder"
                or row.get("split") != "review_unassigned"):
            raise ResearchError("Expected unapproved FOSD research sample; do not relabel its provenance")
        if not isinstance(row.get("text"), str) or unknown_characters(row["text"], cfg.tokenizer):
            raise ResearchError("Unsupported or missing experimental transcript")
        audio = (path.parent / row["audio_path"]).resolve()
        if not audio.is_relative_to(path.parent) or not audio.is_file():
            raise ResearchError("Audio path missing or outside research workspace")
        wave = read_audio(audio, cfg.sample_rate)
        mel = mel_spectrogram(wave, cfg)
        ids = encode(row["text"], cfg.tokenizer)
        if len(mel) > cfg.max_frames or len(mel) <= len(ids) * 2:
            raise ResearchError("Invalid length ratio for weak CTC alignment")
        selected.append({"ids": ids, "mel": mel, "wave": wave,
                         "sha": row.get("source_sha256", ""),
                         "text": row["text"]})
    if not selected:
        raise ResearchError("No real recordings available")
    return selected


def run_research_demo(manifest: str | Path, output_dir: str | Path, *,
                      rights_reviewed: bool = False, ctc_steps: int = 12,
                      acoustic_steps: int = 24, vocoder_steps: int = 10,
                      max_clips: int = 6, device: str = "cpu", seed: int = 7,
                      cfg: Config | None = None) -> dict:
    """Create a native model WAV; report signal validity separately from file validity.

    This experiment uses automatic CTC duration proposals with no human review.
    It is **not** train_acoustic and MUST NOT be promoted to production.
    """
    if not rights_reviewed:
        raise ResearchError("Research training requires explicit source-rights review")
    if not (1 <= ctc_steps <= 200 and 1 <= acoustic_steps <= 400
            and 1 <= vocoder_steps <= 200):
        raise ResearchError("Steps must be positive and bounded")
    cfg = cfg or Config(n_fft=512, n_mels=32, hidden=48,
                        encoder_layers=1, decoder_layers=1, max_frames=1200)
    if cfg.tokenizer != "grapheme":
        raise ResearchError("First research demonstration supports grapheme tokenizer only")
    samples = load_research_samples(manifest, cfg, limit=max_clips)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    target_device = torch.device(device)

    # CTC is trained from random weights on the same unreviewed clips.
    # Its Viterbi timings are weak research guesses, NEVER human-reviewed labels.
    from .aligner import CTCAligner, ctc_viterbi_durations
    ctc = CTCAligner(cfg).to(target_device).train()
    align_opt = torch.optim.AdamW(ctc.parameters(), lr=1e-3)
    ctc_losses = []
    for step in range(ctc_steps):
        sample = samples[step % len(samples)]
        logp = ctc(sample["mel"].to(target_device)[None])[0]
        labels = torch.tensor(sample["ids"], device=target_device, dtype=torch.long)
        loss = F.ctc_loss(logp[:, None, :], labels,
                          torch.tensor([logp.shape[0]], dtype=torch.long),
                          torch.tensor([len(labels)], dtype=torch.long),
                          blank=0, zero_infinity=False)
        if not torch.isfinite(loss):
            raise ResearchError("Nonfinite CTC training loss")
        align_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctc.parameters(), 1.)
        align_opt.step()
        ctc_losses.append(float(loss.detach()))
    ctc.eval()
    for sample in samples:
        with torch.no_grad():
            logp = ctc(sample["mel"].to(target_device)[None])[0]
            durations, _ = ctc_viterbi_durations(logp, sample["ids"])
        if len(durations) != len(sample["ids"]) or sum(durations) != len(sample["mel"]):
            raise ResearchError("CTC proposal length inconsistency")
        sample["durations"] = durations

    acoustic = AcousticModel(cfg, [RESEARCH_VOICE]).to(target_device).train()
    acoustic_opt = torch.optim.AdamW(acoustic.parameters(), lr=1e-3)
    acoustic_losses = []
    for step in range(acoustic_steps):
        sample = samples[step % len(samples)]
        ids = torch.tensor([sample["ids"]], device=target_device)
        durations = torch.tensor([sample["durations"]], device=target_device)
        mel = sample["mel"].to(target_device)[None]
        sid = torch.zeros(1, device=target_device, dtype=torch.long)
        emotion = torch.tensor([EMOTIONS.index("neutral")], device=target_device)
        strength = torch.zeros(1, device=target_device)
        predicted, log_durations, frames = acoustic(ids, sid, emotion, strength, durations)
        loss, _ = acoustic_loss(predicted, mel, frames, log_durations, durations, ids)
        if not torch.isfinite(loss):
            raise ResearchError("Nonfinite acoustic loss")
        acoustic_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(acoustic.parameters(), 1.)
        acoustic_opt.step()
        acoustic_losses.append(float(loss.detach()))

    # Train the project's own randomly initialized generator on short real
    # waveform crops. No external vocoder or voice checkpoint is loaded.
    vocoder = NeuralVocoder(cfg).to(target_device).train()
    vocoder_opt = torch.optim.AdamW(vocoder.parameters(), lr=2e-4)
    vocoder_losses = []
    for step in range(vocoder_steps):
        sample = samples[step % len(samples)]
        frames = len(sample["mel"])
        crop_len = min(48, frames - 4)
        start = 0 if frames == crop_len else (step * 31) % (frames - crop_len)
        mel_crop = sample["mel"][start:start + crop_len].T[None].to(target_device)
        truth = sample["wave"][start * cfg.hop_length:(start + crop_len) * cfg.hop_length]
        truth = F.pad(truth, (0, max(0, crop_len * cfg.hop_length - len(truth))))[:crop_len * cfg.hop_length]
        truth = truth[None].to(target_device)
        generated = vocoder(mel_crop)
        loss = (spectral_loss(generated, truth) + 0.25 * (generated - truth).abs().mean()
                + 0.5 * generated.mean(dim=-1).abs().mean())
        if not torch.isfinite(loss):
            raise ResearchError("Nonfinite vocoder loss")
        vocoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vocoder.parameters(), 1.)
        vocoder_opt.step()
        vocoder_losses.append(float(loss.detach()))

    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ResearchError("Output directory must be empty; do not mix experiments")
    out.mkdir(parents=True, exist_ok=True)
    weights_dir = out / "private_checkpoints"
    weights_dir.mkdir()
    save_acoustic(weights_dir / "acoustic.pt", acoustic.eval(), acoustic_steps)
    save_vocoder(weights_dir / "vocoder.pt", vocoder.eval(), vocoder_steps)
    from .synthesize import NativeSynthesizer
    synth = NativeSynthesizer(weights_dir / "acoustic.pt", weights_dir / "vocoder.pt", device=device)
    # Unseen, fixed text demonstrates TTS inference without teacher durations
    # or replaying a source waveform. Expected to be mostly unintelligible.
    wav = synth.speak("Xin chào, đây là VietDub", RESEARCH_VOICE, emotion="neutral",
                     intensity=0., out_path=out / "vietdub-tts-demo.wav")
    generated, sample_rate = sf.read(wav, dtype="float32")
    duration = len(generated) / sample_rate
    rms = float(np.sqrt(np.mean(generated.astype(np.float64)**2)))
    dc = float(abs(np.mean(generated)))
    ac_rms = float(np.std(generated.astype(np.float64)))
    dc_ratio = dc / max(rms, 1e-10)
    zero_cross_rate = float(np.mean(np.signbit(generated[:-1]) != np.signbit(generated[1:])))
    acoustic_signal_present = (ac_rms >= 0.001 and dc_ratio < 0.90
                               and zero_cross_rate > 0.001)
    if (not math.isfinite(rms) or rms < 1e-5 or duration <= 0.03
            or not np.isfinite(generated).all()):
        raise ResearchError("Generated demo WAV failed signal validation")
    candidate_for_listening = bool(duration >= 2.0 and ac_rms >= 0.01 and dc_ratio < 0.20 and 0.005 < zero_cross_rate < 0.45 and acoustic_signal_present)
    report = {"result": "WAV_SYNTHESIS_SMOKE_ONLY", "intelligibility_verified": False,
              "emotion_verified": False, "speaker_identity_verified": False,
              "training_approved": False, "weak_alignment_reviewed": False,
              "source": EXPECTED_DATASET, "mirror": EXPECTED_MIRROR,
              "clip_count": len(samples),
              "real_audio_seconds": round(sum(len(s["wave"]) for s in samples) / cfg.sample_rate, 3),
              "ctc_steps": ctc_steps, "acoustic_steps": acoustic_steps,
              "vocoder_steps": vocoder_steps,
              "ctc_loss_last": round(ctc_losses[-1], 5),
              "acoustic_loss_last": round(acoustic_losses[-1], 5),
              "vocoder_loss_last": round(vocoder_losses[-1], 5),
              "generated_sample_rate": sample_rate,
              "generated_duration_sec": round(duration, 3),
              "generated_rms": round(rms, 6),
              "generated_ac_rms": round(ac_rms, 6),
              "generated_dc_ratio": round(dc_ratio, 6),
              "generated_zero_crossing_rate": round(zero_cross_rate, 6),
              "acoustic_signal_present": bool(acoustic_signal_present),
              "candidate_for_listening": candidate_for_listening,
              "release_approved": False,
              "source_clip_hashes_sha256": [s["sha"] for s in samples],
              "note": "Independent neural TTS proof-of-generation. Tiny unreviewed real corpus. No realistic Vietnamese speech quality claim; mixed unknown source speakers; checkpoints are not released."}
    (out / "research_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "NOTICE.txt").write_text(
        "Experimental from-scratch Vietnamese TTS demonstration. FPT Open Speech Dataset "
        "(FOSD), original: https://data.mendeley.com/datasets/k9sxg2twv4/4 ; "
        "unofficial viewer mirror: https://huggingface.co/datasets/awghuku/fosd. "
        "Research-only model; no consent, performance quality, speaker identity or "
        "emotion is verified. Do not redistribute as a production voice.\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="VietDub real-audio weak-supervision research demo")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--ctc-steps", type=int, default=12)
    parser.add_argument("--acoustic-steps", type=int, default=24)
    parser.add_argument("--vocoder-steps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--acknowledge-source-rights-reviewed", action="store_true")
    args = parser.parse_args(argv)
    result = run_research_demo(args.manifest, args.output_dir,
                               rights_reviewed=args.acknowledge_source_rights_reviewed,
                               ctc_steps=args.ctc_steps, acoustic_steps=args.acoustic_steps,
                               vocoder_steps=args.vocoder_steps, max_clips=args.max_clips,
                               device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
