"""Train the models and synthesize WAV without external providers."""
from __future__ import annotations
import argparse
import json
from .config import Config
from .symbols import TOKENIZERS
from .dataset import read_manifest
from .train import train_acoustic, train_vocoder
from .synthesize import NativeSynthesizer


def main(argv=None):
    p = argparse.ArgumentParser(prog="vietdub-native", description="Native trainable film-dubbing TTS")
    sub = p.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate-manifest")
    check.add_argument("manifest")
    check.add_argument("--tokenizer", default="grapheme", choices=TOKENIZERS)
    for cmd in ("train-acoustic", "train-vocoder"):
        c = sub.add_parser(cmd)
        c.add_argument("--manifest", required=True)
        c.add_argument("--tokenizer", default="grapheme", choices=TOKENIZERS)
        c.add_argument("--output-dir", required=True)
        c.add_argument("--epochs", type=int, default=1)
        c.add_argument("--device", default="cpu")
        c.add_argument("--lr", type=float, default=2e-4)
        if cmd == "train-acoustic":
            c.add_argument("--batch-size", type=int, default=2)
    s = sub.add_parser("synthesize")
    s.add_argument("--acoustic", required=True)
    s.add_argument("--vocoder", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--speaker", required=True)
    s.add_argument("--emotion", default="neutral")
    s.add_argument("--intensity", type=float, default=.5)
    s.add_argument("--duration-scale", type=float, default=1.0)
    s.add_argument("--device", default="cpu")
    s.add_argument("--output", default="output.wav")
    args = p.parse_args(argv)
    if args.command == "validate-manifest":
        rows = read_manifest(args.manifest, tokenizer=args.tokenizer)
        print(json.dumps({"validated_rows": len(rows), "speakers": sorted({r["speaker_id"] for r in rows})}, ensure_ascii=False))
    elif args.command == "train-acoustic":
        print(train_acoustic(args.manifest, args.output_dir, epochs=args.epochs,
                             batch_size=args.batch_size, device=args.device, lr=args.lr,
                             cfg=Config(tokenizer=args.tokenizer)))
    elif args.command == "train-vocoder":
        print(train_vocoder(args.manifest, args.output_dir, epochs=args.epochs,
                            device=args.device, lr=args.lr,
                            cfg=Config(tokenizer=args.tokenizer)))
    elif args.command == "synthesize":
        path = NativeSynthesizer(args.acoustic, args.vocoder, args.device).speak(
            args.text, args.speaker, args.emotion, args.intensity, args.output, args.duration_scale)
        print(path)
