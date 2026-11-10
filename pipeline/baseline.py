
from statistics import mean, pstdev

BASELINE_SESSIONS = 3
Z_SCORE_THRESHOLD = 2.0
MIN_DRIFT_FEATURES = 2

FEATURES = [
    "average_pitch_hz",
    "pitch_variability_hz",
    "average_rms_energy",
    "speech_duration_seconds",
    "pause_duration_seconds",
    "pause_count",
    "clipping_ratio",
]


def build_baseline(sessions, n=BASELINE_SESSIONS):
    """Build a per-speaker baseline from the first n valid sessions."""
    if n < 2:
        raise ValueError("At least 2 sessions are required for a baseline.")

    if len(sessions) < n:
        raise ValueError(f"Need at least {n} sessions to build a baseline.")

    selected = sessions[:n]
    speaker_ids = {session["speaker_id"] for session in selected}

    if len(speaker_ids) != 1:
        raise ValueError("All baseline sessions must belong to one speaker.")

    baseline_features = {}

    for feature in FEATURES:
        values = [float(session["features"][feature]) for session in selected]
        baseline_features[feature] = {
            "mean": mean(values),
            "std": pstdev(values),
        }

    return {
        "speaker_id": selected[0]["speaker_id"],
        "baseline_sessions": n,
        "features": baseline_features,
    }


def calculate_z_scores(session, baseline):
    """Calculate feature z-scores against a speaker's baseline."""
    if session["speaker_id"] != baseline["speaker_id"]:
        raise ValueError("Session and baseline speaker IDs do not match.")

    z_scores = {}

    for feature in FEATURES:
        value = float(session["features"][feature])
        stats = baseline["features"][feature]
        standard_deviation = stats["std"]

        # Avoid division by zero for features that did not vary at baseline.
        if standard_deviation < 1e-9:
            z_score = 0.0 if abs(value - stats["mean"]) < 1e-9 else (
                float("inf") if value > stats["mean"] else float("-inf")
            )
        else:
            z_score = (value - stats["mean"]) / standard_deviation

        z_scores[feature] = z_score

    return z_scores


def detect_drift(z_scores, threshold=Z_SCORE_THRESHOLD):
    """Flag drift when at least two features exceed the z-score threshold."""
    flagged_features = [
        feature
        for feature, score in z_scores.items()
        if abs(score) >= threshold
    ]

    return {
        "drift_detected": len(flagged_features) >= MIN_DRIFT_FEATURES,
        "flagged_features": flagged_features,
        "flagged_feature_count": len(flagged_features),
    }


def evaluate_session(session, baseline):
    """Calculate z-scores and drift status for one new session."""
    z_scores = calculate_z_scores(session, baseline)
    drift = detect_drift(z_scores)

    return {
        "recording_id": session.get("recording_id"),
        "speaker_id": session["speaker_id"],
        "z_scores": z_scores,
        **drift,
    }
