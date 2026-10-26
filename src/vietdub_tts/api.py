"""FastAPI interface for local or remote VietDub clients."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from .audio import DurationFitError
from .models import DubbingRequest, SpeechRequest
from .service import TTSService


def create_app(service: TTSService | None = None) -> FastAPI:
    app = FastAPI(title="VietDub TTS", version="0.1.0")
    app.state.tts = service or TTSService()

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model_loaded": request.app.state.tts.model_loaded,
                "model_backend": "vieneu-v3-turbo"}

    @app.get("/v1/voices")
    def voices(request: Request):
        profiles = request.app.state.tts.voices.characters
        return {"default_preset": request.app.state.tts.voices.default_preset,
                "characters": [{"speaker_id": key, "voice_id": value.voice_id,
                                "preset": value.preset, "uses_reference": bool(value.ref_audio)}
                               for key, value in profiles.items()]}

    @app.post("/v1/speech")
    def speech(body: SpeechRequest, request: Request):
        try:
            result = request.app.state.tts.synthesize(body)
        except (ValueError, DurationFitError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(result.wav, media_type="audio/wav", headers={
            "X-Voice-ID": result.meta["voice_id"],
            "X-Emotion-Control": result.meta["emotion_control"],
            "X-Audio-Duration": str(result.meta["duration"]),
        })

    @app.post("/v1/dubbing/segments")
    def dubbing(body: DubbingRequest, request: Request):
        try:
            blob = request.app.state.tts.dub_segments(body)
        except (ValueError, DurationFitError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(blob, media_type="application/zip", headers={
            "Content-Disposition": 'attachment; filename="vietdub-segments.zip"',
        })

    return app


app = create_app()
