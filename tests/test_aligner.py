"""Native CTC aligner: structural regression tests, not speech-quality benchmarks."""
import json
import numpy as np
import pytest
import torch

from vietdub_native.aligner import (
    AlignmentError, TranscriptDataset, CTCAligner, ctc_viterbi_durations,
    collate_transcripts, ctc_loss, save_aligner, load_aligner,
    train_aligner, propose_alignments, read_transcripts,
)
from vietdub_native.dataset import AlignedSpeechDataset, ManifestError


def test_viterbi_blanks_and_durations():
    scores = np.full((7, 4), -11.0)
    scores[[0, 3, 6], 0] = -.01
    scores[[1, 2], 2] = -.01
    scores[[4, 5], 3] = -.01
    durations, confidence = ctc_viterbi_durations(scores, [2, 3])
    assert len(durations) == 2
    assert min(durations) > 0 and sum(durations) == 7
    assert 0.9 < confidence <= 1


def test_repeated_ctc_token_needs_blank():
    scores = np.full((3, 4), -12.0)
    scores[0, 2] = scores[1, 0] = scores[2, 2] = -.01
    durations, _ = ctc_viterbi_durations(scores, [2, 2])
    assert min(durations) > 0 and sum(durations) == 3
    with pytest.raises(AlignmentError, match="more frames"):
        ctc_viterbi_durations(scores[:2], [2, 2])


def test_reject_unreachable_and_nonfinite():
    with pytest.raises(AlignmentError, match="finite"):
        ctc_viterbi_durations(np.full((4, 4), np.nan), [2])
    for invalid in ([0], [4]):
        with pytest.raises(AlignmentError, match="token IDs"):
            ctc_viterbi_durations(np.ones((4, 4)), invalid)


def test_transcript_without_durations(tiny_dataset):
    manifest, cfg = tiny_dataset
    row = json.loads(manifest.read_text())
    row.pop("token_durations")
    manifest.write_text(json.dumps(row, ensure_ascii=False))
    ds = TranscriptDataset(manifest, cfg)
    batch = collate_transcripts([ds[0], ds[0]])
    assert batch["mels"].shape[0] == 2
    assert batch["tokens"].numel() == 2 * len(ds[0]["tokens"])


def test_ctc_backward_gradient(tiny_dataset):
    manifest, cfg = tiny_dataset
    batch = collate_transcripts([TranscriptDataset(manifest, cfg)[0]])
    model = CTCAligner(cfg)
    loss = ctc_loss(model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None


def test_train_checkpoint_proposals_and_review_gate(tiny_dataset, tmp_path):
    manifest, cfg = tiny_dataset
    checkpoint = tmp_path / "aligner.pt"
    history = train_aligner(manifest, checkpoint, cfg=cfg, epochs=1)
    assert len(history) == 1 and np.isfinite(history[0]["ctc_loss"])
    assert load_aligner(checkpoint).cfg == cfg
    output = tmp_path / "review" / "proposed.jsonl"
    proposals = propose_alignments(manifest, output, checkpoint)
    assert len(proposals) == 1
    row = proposals[0]
    assert row["alignment_verified"] is False
    assert row["alignment_status"] == "needs_human_review"
    assert row["alignment_method"].startswith("native_ctc_v1:")
    assert sum(row["token_durations"]) == len(TranscriptDataset(manifest, cfg)[0]["mel"])
    with pytest.raises(ManifestError, match="review"):
        AlignedSpeechDataset(output, cfg, ["actor_a"])
    row["alignment_verified"] = True
    output.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert len(AlignedSpeechDataset(output, cfg, ["actor_a"])[0]["mel"]) > 0


def test_refuse_input_overwrite(tiny_dataset, tmp_path):
    manifest, cfg = tiny_dataset
    checkpoint = tmp_path / "aligner.pt"
    save_aligner(checkpoint, CTCAligner(cfg))
    with pytest.raises(AlignmentError, match="overwrite"):
        propose_alignments(manifest, manifest, checkpoint)


def test_checkpoint_exact_vocab(tiny_dataset, tmp_path):
    _, cfg = tiny_dataset
    checkpoint = tmp_path / "aligner.pt"
    save_aligner(checkpoint, CTCAligner(cfg))
    altered = torch.load(checkpoint, weights_only=True)
    altered["vocab"][3] = "changed"
    torch.save(altered, checkpoint)
    with pytest.raises(AlignmentError, match="vocabulary mismatch"):
        load_aligner(checkpoint)


def test_invalid_text_and_tokenizer(tiny_dataset):
    manifest, cfg = tiny_dataset
    row = json.loads(manifest.read_text())
    row["text"] = "中文"
    manifest.write_text(json.dumps(row, ensure_ascii=False))
    with pytest.raises(AlignmentError, match="Unsupported"):
        read_transcripts(manifest, cfg)
    row["text"] = "xin chào"
    row["tokenizer"] = "vi_phonetic_v1"
    manifest.write_text(json.dumps(row, ensure_ascii=False))
    with pytest.raises(AlignmentError, match="Tokenizer mismatch"):
        read_transcripts(manifest, cfg)
