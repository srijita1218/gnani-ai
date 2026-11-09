"""Thin wrapper around the FastAPI backend, used by the Streamlit UI."""
from __future__ import annotations

import requests

DEFAULT_API_URL = "http://localhost:8000"
TIMEOUT_S = 10
AUDIO_TIMEOUT_S = 180   # speech-to-text plus feature extraction can take a while


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _check(response: requests.Response):
    if response.ok:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise ApiError(response.status_code, str(detail)[:500])


def _call(method: str, url: str, **kwargs):
    try:
        return _check(requests.request(method, url, **kwargs))
    except requests.exceptions.RequestException as exc:
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            raise ApiError(0, f"Cannot reach the backend at {url.split('/')[0]}//{url.split('/')[2]}. "
                              "Start it with: uvicorn app.main:app --port 8000") from exc
        raise ApiError(0, str(exc)) from exc


def health(base_url: str) -> dict:
    return _call("GET", f"{base_url}/health", timeout=TIMEOUT_S)


def create_participant(base_url: str, participant_id: str) -> dict:
    return _call(
        "POST", f"{base_url}/patients", timeout=TIMEOUT_S,
        json={"participant_id": participant_id, "display_name": participant_id, "consent_given": True},
    )


def get_sessions(base_url: str, participant_id: str) -> list:
    return _call("GET", f"{base_url}/patients/{participant_id}/sessions", timeout=TIMEOUT_S)


def post_audio(base_url: str, participant_id: str, language: str, filename: str, data: bytes) -> dict:
    return _call(
        "POST", f"{base_url}/sessions/audio", timeout=AUDIO_TIMEOUT_S,
        params={"participant_id": participant_id, "language": language, "filename": filename},
        data=data, headers={"Content-Type": "application/octet-stream"},
    )
