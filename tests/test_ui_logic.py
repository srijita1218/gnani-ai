"""Offline tests for the UI's logic and mock data (no Streamlit needed)."""
import json
import math
from pathlib import Path

from app.ui.logic import (
    FEATURE_KEYS,
    LOW_MAX,
    MODERATE_MAX,
    badge_html,
    compute_baseline,
    latest_risk_score,
    risk_level,
    sessions_to_frame,
)
from app.ui.mock_data import MOCK_PATIENTS, N_SESSIONS

CONTRACT = Path(__file__).resolve().parent.parent / "contract" / "mock_session.json"


def test_risk_level_thresholds():
    assert risk_level(0.0) == "low"
    assert risk_level(LOW_MAX - 0.001) == "low"
    assert risk_level(LOW_MAX) == "moderate"
    assert risk_level(MODERATE_MAX - 0.001) == "moderate"
    assert risk_level(MODERATE_MAX) == "high"
    assert risk_level(1.0) == "high"


def test_risk_level_missing_or_bad_values_are_unknown():
    for bad in (None, float("nan"), "abc", True):
        assert risk_level(bad) == "unknown"


def test_badge_colours_and_text():
    assert "#2e7d32" in badge_html("low", 0.1) and "Low" in badge_html("low", 0.1)
    assert "#f9a825" in badge_html("moderate", 0.5)
    assert "#c62828" in badge_html("high", 0.9) and "0.90" in badge_html("high", 0.9)
    assert "No risk score yet" in badge_html("unknown")


def test_mock_patients_cover_all_three_badges():
    levels = {}
    for pid, patient in MOCK_PATIENTS.items():
        df = sessions_to_frame(patient["sessions"])
        levels[pid] = risk_level(latest_risk_score(df))
    assert levels == {"P001": "low", "P002": "moderate", "P003": "high"}


def test_mock_sessions_use_the_contract_feature_names():
    contract_keys = set(json.loads(CONTRACT.read_text(encoding="utf-8"))["features"])
    assert set(FEATURE_KEYS) == contract_keys
    for patient in MOCK_PATIENTS.values():
        assert len(patient["sessions"]) == N_SESSIONS
        for session in patient["sessions"]:
            assert set(session["features"]) == contract_keys
            assert 0.0 <= session["risk"]["score"] <= 1.0


def test_mock_data_is_deterministic():
    from app.ui import mock_data
    first = mock_data._build_sessions("P002", mock_data._PATIENTS["P002"])
    second = mock_data._build_sessions("P002", mock_data._PATIENTS["P002"])
    assert first == second


def test_frame_is_sorted_oldest_first_even_if_input_is_shuffled():
    sessions = list(reversed(MOCK_PATIENTS["P002"]["sessions"]))
    df = sessions_to_frame(sessions)
    assert list(df["timestamp"]) == sorted(df["timestamp"])
    assert df["session_id"].iloc[0] == "P002_S01"


def test_frame_handles_sessions_without_risk_or_features():
    df = sessions_to_frame([{"session_id": "x", "timestamp": "2026-11-08T18:00:00+05:30"}])
    assert len(df) == 1
    assert math.isnan(df["risk_score"].iloc[0])
    assert latest_risk_score(df) is None
    assert risk_level(latest_risk_score(df)) == "unknown"


def test_baseline_is_mean_and_std_of_first_three_sessions():
    df = sessions_to_frame(MOCK_PATIENTS["P003"]["sessions"])
    b = compute_baseline(df, "speaking_rate")
    first3 = df["speaking_rate"].iloc[:3]
    assert abs(b["mean"] - first3.mean()) < 1e-9
    assert abs(b["std"] - first3.std(ddof=0)) < 1e-9
    assert b["n"] == 3


def test_baseline_needs_enough_sessions():
    df = sessions_to_frame(MOCK_PATIENTS["P001"]["sessions"][:2])
    assert compute_baseline(df, "speaking_rate") is None


def test_declining_patient_trends_away_from_baseline():
    df = sessions_to_frame(MOCK_PATIENTS["P003"]["sessions"])
    b = compute_baseline(df, "speaking_rate")
    assert df["speaking_rate"].iloc[-1] < b["mean"] - b["std"]       # slower than baseline
    b = compute_baseline(df, "pause_ratio")
    assert df["pause_ratio"].iloc[-1] > b["mean"] + b["std"]         # more pauses than baseline
