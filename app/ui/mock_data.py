"""Deterministic demo patients for the UI (no backend needed).

Three patients with 8 weekly sessions each, in the contract's session shape
plus a `risk` block ({"score": 0..1}) that the real contract does not have yet.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
FIRST_SESSION = datetime(2026, 10, 5, 18, 0, tzinfo=IST)
N_SESSIONS = 8

BASE_FEATURES = {
    "speaking_rate": 2.9,
    "mean_pause_duration": 0.35,
    "pitch_mean": 145.0,
    "pitch_variation": 24.0,
    "rms_energy": 0.18,
    "pause_ratio": 0.20,
    "energy_variation": 0.55,
    "jitter_local": 0.010,
    "shimmer_local": 0.050,
    "hnr_db": 18.0,
}

# fractional change per session (negative = falls over time)
_STABLE = {}
_DRIFTING = {
    "speaking_rate": -0.025, "mean_pause_duration": 0.04, "pause_ratio": 0.035,
    "hnr_db": -0.012, "jitter_local": 0.03, "pitch_variation": -0.02,
}
_DECLINING = {
    "speaking_rate": -0.045, "mean_pause_duration": 0.07, "pause_ratio": 0.065,
    "hnr_db": -0.025, "jitter_local": 0.06, "shimmer_local": 0.05,
    "pitch_variation": -0.04, "energy_variation": -0.03,
}

_PATIENTS = {
    "P001": {
        "label": "P001 · stable (example)",
        "pitch_mean": 120.0, "drift": _STABLE,
        "risk": [0.08, 0.10, 0.07, 0.12, 0.09, 0.11, 0.10, 0.08],
    },
    "P002": {
        "label": "P002 · gradual drift (example)",
        "pitch_mean": 190.0, "drift": _DRIFTING,
        "risk": [0.10, 0.12, 0.18, 0.28, 0.38, 0.46, 0.52, 0.58],
    },
    "P003": {
        "label": "P003 · marked change (example)",
        "pitch_mean": 165.0, "drift": _DECLINING,
        "risk": [0.12, 0.15, 0.30, 0.45, 0.62, 0.71, 0.80, 0.86],
    },
}


def _build_sessions(pid: str, spec: dict) -> list:
    rng = random.Random(pid)   # string seed -> same data on every run
    base = dict(BASE_FEATURES, pitch_mean=spec["pitch_mean"])
    sessions = []
    for i in range(N_SESSIONS):
        features = {}
        for key, value in base.items():
            drift = 1 + spec["drift"].get(key, 0.0) * i
            noise = 1 + rng.gauss(0, 0.02)
            features[key] = round(value * drift * noise, 5)
        features["pause_ratio"] = round(min(features["pause_ratio"], 0.95), 5)
        score = spec["risk"][i]
        sessions.append({
            "schema_version": "1.1",
            "session_id": f"{pid}_S{i + 1:02d}",
            "participant_id": pid,
            "timestamp": (FIRST_SESSION + timedelta(weeks=i)).isoformat(),
            "features": features,
            "baseline": {"available": i >= 3, "recordings_used": min(i, 3)},
            "monitoring_status": "insufficient_baseline" if i < 3 else (
                "within_baseline" if score < 0.33 else "deviation_detected"
            ),
            "risk": {"score": score},
            "metadata": {"data_type": "mock", "is_clinically_validated": False},
        })
    return sessions


MOCK_PATIENTS = {
    pid: {"label": spec["label"], "sessions": _build_sessions(pid, spec)}
    for pid, spec in _PATIENTS.items()
}
