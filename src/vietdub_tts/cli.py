"""Run the service or synthesize one Vietnamese film dialogue line."""
from __future__ import annotations

import argparse
from pathlib import Path

from .models import Emotion, SpeechRequest
from .service import TTSService


def main() -> None:
    parser = argparse.ArgumentParser(prog="vietdub-tts")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    say = sub.add_parser("speak")
    say.add_argument("text")
    say.add_argument("--voice", default=None)
    say.add_argument("--emotion", default="neutral", choices=[e.value for e in Emotion])
    say.add_argument("--output", default="output.wav")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run("vietdub_tts.api:app", host=args.host, port=args.port)
    else:
        audio = TTSService().synthesize(SpeechRequest(
            text=args.text, voice_id=args.voice, emotion=Emotion(args.emotion)))
        Path(args.output).write_bytes(audio.wav)
        print(f"WAV saved: {args.output} | emotion={audio.meta['emotion_control']}")


if __name__ == "__main__":
    main()
