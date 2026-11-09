"""Tests for POST /sessions/audio. The slow audio pipeline is replaced by a fake,
so these run offline and need only FastAPI."""
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app

URL = "/sessions/audio?participant_id=P001&language=en-IN&filename=a.wav"
HEADERS = {"Content-Type": "application/octet-stream"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def make_patient(client, pid="P001", consent=True):
    r = client.post("/patients", json={"participant_id": pid, "consent_given": consent})
    assert r.status_code == 201


def fake_session(path, participant_id, language, filename):
    return {
        "schema_version": "1.1",
        "session_id": "SESSION_AUDIO1",
        "participant_id": participant_id,
        "timestamp": "2026-11-09T10:00:00+05:30",
        "features": {"speaking_rate": 2.0},
        "transcription": {"language": language, "text": "hello", "status": "completed"},
        "recording": {"filename": filename},
    }


def test_audio_is_processed_saved_and_listed(client, monkeypatch):
    seen = {}

    def fake(path, participant_id, language, filename):
        seen["path"], seen["bytes"] = path, path.read_bytes()
        return fake_session(path, participant_id, language, filename)

    monkeypatch.setattr(main_module, "build_session_from_audio", fake)
    make_patient(client)

    r = client.post(URL, content=b"RIFFfakeaudio", headers=HEADERS)
    assert r.status_code == 201
    assert r.json()["session_id"] == "SESSION_AUDIO1"
    assert seen["bytes"] == b"RIFFfakeaudio"
    assert not seen["path"].exists()                      # temp file deleted afterwards

    listed = client.get("/patients/P001/sessions").json()
    assert [s["session_id"] for s in listed] == ["SESSION_AUDIO1"]
    assert listed[0]["transcription"]["language"] == "en-IN"


def test_unknown_participant_is_404(client):
    assert client.post(URL, content=b"x", headers=HEADERS).status_code == 404


def test_participant_without_consent_is_403(client):
    make_patient(client, consent=False)
    assert client.post(URL, content=b"x", headers=HEADERS).status_code == 403


def test_empty_body_is_400(client):
    make_patient(client)
    assert client.post(URL, content=b"", headers=HEADERS).status_code == 400


def test_unsupported_file_type_is_415(client):
    make_patient(client)
    r = client.post(
        "/sessions/audio?participant_id=P001&filename=notes.txt", content=b"x", headers=HEADERS
    )
    assert r.status_code == 415


def test_bad_audio_is_422_and_nothing_is_saved(client, monkeypatch):
    def fake(*args):
        raise ValueError("Recording too short (0.20s); minimum is 0.5s")

    monkeypatch.setattr(main_module, "build_session_from_audio", fake)
    make_patient(client)
    r = client.post(URL, content=b"x", headers=HEADERS)
    assert r.status_code == 422
    assert "too short" in r.json()["detail"]
    assert client.get("/patients/P001/sessions").json() == []


def test_missing_dependencies_is_503(client, monkeypatch):
    def fake(*args):
        raise RuntimeError("Feature extraction dependencies are missing.")

    monkeypatch.setattr(main_module, "build_session_from_audio", fake)
    make_patient(client)
    assert client.post(URL, content=b"x", headers=HEADERS).status_code == 503
