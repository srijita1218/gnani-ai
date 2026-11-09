import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

MOCK_PATH = Path(__file__).resolve().parent.parent / "contract" / "mock_session.json"


def load_mock() -> dict:
    return json.loads(MOCK_PATH.read_text())


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:   # `with` runs startup -> creates tables
        yield c


def make_patient(client, participant_id, consent=True):
    r = client.post(
        "/patients",
        json={"participant_id": participant_id, "consent_given": consent},
    )
    assert r.status_code == 201
    return r.json()["id"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_post_mock_session_and_read_back(client):
    mock = load_mock()
    pid = make_patient(client, mock["participant_id"])

    r = client.post("/sessions", json=mock)       # mock posted unchanged
    assert r.status_code == 201

    r = client.get(f"/patients/{pid}/sessions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0] == mock                        # nested fields survive round-trip


def test_duplicate_patient_rejected(client):
    make_patient(client, "P001")
    r = client.post("/patients", json={"participant_id": "P001", "consent_given": True})
    assert r.status_code == 409


def test_session_requires_existing_patient(client):
    assert client.post("/sessions", json=load_mock()).status_code == 404


def test_session_requires_consent(client):
    mock = load_mock()
    make_patient(client, mock["participant_id"], consent=False)
    assert client.post("/sessions", json=mock).status_code == 403


def test_duplicate_session_rejected(client):
    mock = load_mock()
    make_patient(client, mock["participant_id"])
    assert client.post("/sessions", json=mock).status_code == 201
    assert client.post("/sessions", json=mock).status_code == 409


def test_sessions_ordered_oldest_first(client):
    mock = load_mock()
    pid = make_patient(client, mock["participant_id"])
    for i, ts in enumerate(["2026-11-09T10:00:00+05:30", "2026-11-08T10:00:00+05:30"]):
        s = dict(mock, session_id=f"S{i}", timestamp=ts)
        assert client.post("/sessions", json=s).status_code == 201
    got = [x["timestamp"] for x in client.get(f"/patients/{pid}/sessions").json()]
    assert got == sorted(got)
