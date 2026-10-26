"""Optional self-hosted API backed ONLY by native checkpoints."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from .synthesize import NativeSynthesizer


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    speaker_id: str
    emotion: str = "neutral"
    emotion_intensity: float = Field(default=.5, ge=0, le=1)
    duration_scale: float = Field(default=1., ge=.8, le=1.2)


def create_app(synthesizer=None):
    app = FastAPI(title="VietDub Native Expressive TTS", version="0.1.0")
    state = {"synth": synthesizer}

    def get_synth():
        if state["synth"] is None:
            acoustic = os.environ.get("VIETDUB_ACOUSTIC_CHECKPOINT")
            vocoder = os.environ.get("VIETDUB_VOCODER_CHECKPOINT")
            if not acoustic or not vocoder:
                raise HTTPException(503, "Native checkpoints not configured or trained")
            try:
                state["synth"] = NativeSynthesizer(acoustic, vocoder)
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(503, f"Unable to load checkpoints: {exc}") from exc
        return state["synth"]

    @app.get("/health")
    def health():
        return {"status": "ready" if state["synth"] is not None else "needs_checkpoints", "model": "vietdub_native_v1"}

    @app.get("/v1/voices")
    def voices():
        model = get_synth().acoustic
        return {"speakers": model.speakers, "emotions": model.emotions, "emotion_control": "experimental"}

    @app.post("/v1/speech")
    def speech(req: SpeechRequest):
        synth = get_synth()
        fd, temp = tempfile.mkstemp(prefix="vietdub_native_", suffix=".wav")
        os.close(fd)
        try:
            synth.speak(req.text, req.speaker_id, req.emotion, req.emotion_intensity,
                        temp, req.duration_scale)
        except (ValueError, RuntimeError) as e:
            Path(temp).unlink(missing_ok=True)
            raise HTTPException(422, str(e)) from e
        except Exception:
            Path(temp).unlink(missing_ok=True)
            raise
        return FileResponse(temp, media_type="audio/wav", filename="speech.wav",
                            background=BackgroundTask(lambda: Path(temp).unlink(missing_ok=True)))
    return app


app = create_app()
