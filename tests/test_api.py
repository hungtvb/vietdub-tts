from fastapi.testclient import TestClient
from vietdub_native.api import create_app


def test_api_needs_native_checkpoints():
    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "needs_checkpoints"
    assert client.post("/v1/speech", json={"text": "xin chào", "speaker_id": "a"}).status_code == 503


def test_api_validates_intensity_before_model():
    client = TestClient(create_app())
    assert client.post("/v1/speech", json={"text": "xin chào", "speaker_id": "a", "emotion_intensity": 1.5}).status_code == 422
