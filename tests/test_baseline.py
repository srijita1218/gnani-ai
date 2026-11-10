
import pytest

from pipeline.baseline import build_baseline, evaluate_session


def make_session(recording_id, speaker_id, values):
    return {
        "recording_id": recording_id,
        "speaker_id": speaker_id,
        "features": values,
    }


def make_features(
    pitch=150,
    pitch_variability=20,
    energy=0.05,
    duration=10,
    pause_duration=1,
    pause_count=3,
    clipping=0.01,
):
    return {
        "average_pitch_hz": pitch,
        "pitch_variability_hz": pitch_variability,
        "average_rms_energy": energy,
        "speech_duration_seconds": duration,
        "pause_duration_seconds": pause_duration,
        "pause_count": pause_count,
        "clipping_ratio": clipping,
    }


def test_slowing_patient_triggers_drift_at_expected_session():
    patient_sessions = [
        make_session("slow-1", "patient-slow", make_features(duration=10, pause_duration=1, pause_count=3)),
        make_session("slow-2", "patient-slow", make_features(duration=11, pause_duration=1.1, pause_count=3)),
        make_session("slow-3", "patient-slow", make_features(duration=9, pause_duration=0.9, pause_count=2)),
        make_session("slow-4", "patient-slow", make_features(duration=10.5, pause_duration=1.05, pause_count=3)),
        make_session("slow-5", "patient-slow", make_features(duration=18, pause_duration=4, pause_count=9)),
    ]

    baseline = build_baseline(patient_sessions, n=3)

    # Session 4 should remain below the drift threshold.
    session_4_result = evaluate_session(patient_sessions[3], baseline)
    assert session_4_result["drift_detected"] is False

    # Session 5 has a substantial slowdown and should trigger drift.
    session_5_result = evaluate_session(patient_sessions[4], baseline)
    assert session_5_result["drift_detected"] is True
    assert "speech_duration_seconds" in session_5_result["flagged_features"]
    assert "pause_duration_seconds" in session_5_result["flagged_features"]
    assert "pause_count" in session_5_result["flagged_features"]


def test_stable_patient_never_triggers_drift():
    patient_sessions = [
        make_session("stable-1", "patient-stable", make_features(duration=10, pause_duration=1, pause_count=3)),
        make_session("stable-2", "patient-stable", make_features(duration=11, pause_duration=1.1, pause_count=3)),
        make_session("stable-3", "patient-stable", make_features(duration=9, pause_duration=0.9, pause_count=2)),
        make_session("stable-4", "patient-stable", make_features(duration=10.5, pause_duration=1.05, pause_count=3)),
        make_session("stable-5", "patient-stable", make_features(duration=9.5, pause_duration=0.95, pause_count=2)),
    ]

    baseline = build_baseline(patient_sessions, n=3)

    for session in patient_sessions[3:]:
        result = evaluate_session(session, baseline)
        assert result["drift_detected"] is False


def test_baseline_rejects_mixed_speakers():
    sessions = [
        make_session("a", "patient-a", make_features()),
        make_session("b", "patient-b", make_features()),
        make_session("c", "patient-a", make_features()),
    ]

    with pytest.raises(ValueError):
        build_baseline(sessions, n=3)


def test_baseline_requires_enough_sessions():
    sessions = [
        make_session("a", "patient-a", make_features()),
        make_session("b", "patient-a", make_features()),
    ]

    with pytest.raises(ValueError):
        build_baseline(sessions, n=3)
