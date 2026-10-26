"""Per-sentence speech and batch film-dubbing artifact generation."""
from __future__ import annotations

import io
import json
import os
import threading
import zipfile
from dataclasses import dataclass

from .audio import render_wav
from .emotion import plan_emotion
from .engine import SpeechEngine, VieNeuEngine
from .models import DubbingRequest, SpeechRequest
from .voices import VoiceRegistry


@dataclass(frozen=True)
class SpeechResult:
    wav: bytes
    meta: dict


class TTSService:
    def __init__(self, engine: SpeechEngine | None = None,
                 voices: VoiceRegistry | None = None):
        self._engine = engine
        self._lock = threading.Lock()
        voice_file = os.environ.get("VIETDUB_TTS_VOICES_FILE")
        self.voices = voices or (VoiceRegistry.from_json(voice_file)
                                if voice_file else VoiceRegistry())

    @property
    def model_loaded(self) -> bool:
        return self._engine is not None

    def _get_engine(self) -> SpeechEngine:
        # Keep the default model lazy; CPU model initialization is expensive.
        with self._lock:
            if self._engine is None:
                self._engine = VieNeuEngine()
            return self._engine

    def synthesize(self, req: SpeechRequest) -> SpeechResult:
        voice = self.voices.resolve(req.speaker_id, req.voice_id)
        plan = plan_emotion(req.text.strip(), req.emotion, req.emotion_intensity)
        # Serialize model inference; do not assume the backend is thread-safe.
        engine = self._get_engine()
        with self._lock:
            generated = engine.infer(plan.text_for_model, voice)
        wav, duration = render_wav(generated, req.target_duration)
        meta = {
            "voice_id": voice.voice_id,
            "speaker_id": req.speaker_id,
            "emotion_requested": req.emotion.value,
            "emotion_control": plan.control_level,
            "emotion_note": plan.note,
            "duration": duration,
            "sample_rate": 44100,
            "format": "wav_pcm16_mono",
        }
        return SpeechResult(wav, meta)

    def dub_segments(self, request: DubbingRequest) -> bytes:
        output = io.BytesIO()
        manifest = {"schema_version": 1, "sample_rate": 44100, "segments": []}
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(request.segments, key=lambda s: (s.start, s.index)):
                # Original source timestamps are authoritative for the dub slot.
                req = SpeechRequest(**item.model_dump(exclude={"index", "start", "end",
                                                            "target_duration"}),
                                    target_duration=item.end - item.start)
                generated = self.synthesize(req)
                name = f"seg_{item.index:04d}.wav"
                archive.writestr(name, generated.wav)
                manifest["segments"].append({"index": item.index, "start": item.start,
                                             "end": item.end, "audio_file": name,
                                             **generated.meta})
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        return output.getvalue()
