
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "referrals.db"))
    with TestClient(app) as test_client:
        yield test_client


def create_patient(client):
    response = client.post(
        "/patients",
        json={
            "participant_id": "P001",
            "consent_given": True,
        },
    )
    assert response.status_code == 201


def make_session(score, language="en-IN"):
    return {
        "session_id": f"SESSION_{score}",
        "participant_id": "P001",
        "timestamp": "2026-11-10T10:00:00+05:30",
        "transcription": {
            "language": language,
            "text": "Sample speech",
        },
        "features": {
            "speaking_rate": 1.2,
        },
        "risk": {
            "score": score,
            "domain": "Speech monitoring",
        },
    }


def test_high_risk_session_creates_referral(client):
    create_patient(client)

    response = client.post("/sessions", json=make_session(0.80))
    assert response.status_code == 201

    queue = client.get("/referrals")
    assert queue.status_code == 200
    assert len(queue.json()) == 1

    referral = queue.json()[0]
    assert referral["patient_id"] == "P001"
    assert referral["domain"] == "Speech monitoring"
    assert referral["score"] == 0.80
    assert referral["status"] == "Pending"


def test_below_threshold_does_not_create_referral(client):
    create_patient(client)

    response = client.post("/sessions", json=make_session(0.65))
    assert response.status_code == 201
    assert client.get("/referrals").json() == []


def test_threshold_score_creates_referral_and_generates_notes(client):
    create_patient(client)

    response = client.post(
        "/sessions",
        json=make_session(0.66, language="hi-IN"),
    )
    assert response.status_code == 201

    queue = client.get("/referrals").json()
    assert len(queue) == 1

    referral = queue[0]
    assert referral["patient_language"] == "hi-IN"
    assert referral["clinician_note"]
    assert "not a diagnosis" in referral["clinician_note"].lower()
    assert referral["patient_message"]
    assert "निदान" in referral["patient_message"]
