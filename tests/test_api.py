import io
import json
import zipfile

import soundfile as sf
from fastapi.testclient import TestClient

from vietdub_tts.api import create_app


def test_health_does_not_download_model(test_service):
    client = TestClient(create_app(test_service))
    result = client.get("/health")
    assert result.status_code == 200
    assert result.json()["model_loaded"] is True  # injected fake engine


def test_speech_returns_wav_and_honest_emotion_status(test_service):
    client = TestClient(create_app(test_service))
    result = client.post("/v1/speech", json={
        "text": "Anh đừng đi", "speaker_id": "heroine", "emotion": "crying"})
    assert result.status_code == 200
    assert result.headers["content-type"] == "audio/wav"
    assert result.headers["x-emotion-control"] == "uncontrolled"
    assert sf.info(io.BytesIO(result.content)).samplerate == 44100


def test_dubbing_zip_preserves_indices_timeline_and_character(test_service):
    client = TestClient(create_app(test_service))
    result = client.post("/v1/dubbing/segments", json={"segments": [
        {"index": 2, "start": 1.5, "end": 2.5, "text": "Em đây", "speaker_id": "heroine"},
        {"index": 1, "start": 0, "end": 1.0, "text": "Anh đây", "speaker_id": "hero"},
    ]})
    assert result.status_code == 200, result.text
    with zipfile.ZipFile(io.BytesIO(result.content)) as z:
        assert set(z.namelist()) == {"seg_0001.wav", "seg_0002.wav", "manifest.json"}
        meta = json.loads(z.read("manifest.json"))
        assert [x["index"] for x in meta["segments"]] == [1, 2]
        assert [x["voice_id"] for x in meta["segments"]] == ["hero", "heroine"]
        for item in meta["segments"]:
            assert item["duration"] == 1.0
            assert sf.info(io.BytesIO(z.read(item["audio_file"]))).samplerate == 44100


def test_unknown_speaker_fails_not_fallback(test_service):
    client = TestClient(create_app(test_service))
    result = client.post("/v1/speech", json={"text": "Em đây", "speaker_id": "???"})
    assert result.status_code == 422
    assert "unknown speaker" in result.text


def test_duplicate_index_rejected(test_service):
    client = TestClient(create_app(test_service))
    segment = {"index": 0, "start": 0, "end": 1, "speaker_id": "hero", "text": "Xin chào"}
    result = client.post("/v1/dubbing/segments", json={"segments": [segment, segment]})
    assert result.status_code == 422


def test_invalid_timestamps_rejected(test_service):
    client = TestClient(create_app(test_service))
    result = client.post("/v1/dubbing/segments", json={"segments": [
        {"index": 0, "start": 2, "end": 1, "speaker_id": "hero", "text": "Hi"}]})
    assert result.status_code == 422
