"""UI logic with no Streamlit import, so it can be tested offline."""
from __future__ import annotations

import html
import math

import pandas as pd

# Feature keys and labels, in the same order as the contract's features block.
FEATURE_LABELS = {
    "speaking_rate": "Speaking rate (words/s)",
    "mean_pause_duration": "Mean pause length (s)",
    "pitch_mean": "Mean pitch (Hz)",
    "pitch_variation": "Pitch variation (Hz)",
    "rms_energy": "Loudness (RMS)",
    "pause_ratio": "Pause ratio (fraction of clip)",
    "energy_variation": "Loudness variation",
    "jitter_local": "Jitter (fraction)",
    "shimmer_local": "Shimmer (fraction)",
    "hnr_db": "Harmonics-to-noise ratio (dB)",
}
FEATURE_KEYS = tuple(FEATURE_LABELS)

# Label shown in the picker -> language_code sent to Gnani Prisma.
# "indic-auto" is the default already used by backend/stt.py; add more codes here.
LANGUAGES = {
    "Auto-detect (Indian languages)": "indic-auto",
    "English (India)": "en-IN",
    "Hindi": "hi-IN",
}
DEFAULT_LANGUAGE = "indic-auto"

# risk.score is assumed to be 0..1. These cut-offs are placeholders until the
# risk-scoring step defines the real scale; change them here only.
LOW_MAX = 0.33        # score below this  -> green
MODERATE_MAX = 0.66   # score below this  -> amber, otherwise red

BASELINE_SESSIONS = 3  # first N sessions define the personal baseline in the chart

# level -> (label, background, text colour)
RISK_STYLES = {
    "low": ("Low", "#2e7d32", "#ffffff"),
    "moderate": ("Moderate", "#f9a825", "#1b1b1b"),
    "high": ("High", "#c62828", "#ffffff"),
    "unknown": ("No risk score yet", "#757575", "#ffffff"),
}
RISK_MESSAGES = {
    "low": "Recent speech is close to this person's usual pattern.",
    "moderate": "Some speech features have drifted from this person's baseline. Keep monitoring.",
    "high": "Speech features have moved well away from baseline. Consider a follow-up with a healthcare professional.",
    "unknown": "Not enough recordings or no risk score available yet.",
}


def risk_level(score) -> str:
    """'low' | 'moderate' | 'high' | 'unknown' from a risk.score."""
    if score is None or isinstance(score, bool):
        return "unknown"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(score):
        return "unknown"
    if score < LOW_MAX:
        return "low"
    if score < MODERATE_MAX:
        return "moderate"
    return "high"


def badge_html(level: str, score=None) -> str:
    """Coloured pill. The text carries the level too, so colour is never the only signal."""
    label, background, text = RISK_STYLES.get(level, RISK_STYLES["unknown"])
    suffix = f" · {float(score):.2f}" if score is not None and level != "unknown" else ""
    return (
        f'<span style="background:{background};color:{text};padding:6px 18px;'
        f'border-radius:999px;font-weight:700;font-size:1.1rem;display:inline-block">'
        f"{html.escape(label)}{suffix}</span>"
    )


SESSION_COLUMNS = ["session_id", "timestamp", "monitoring_status", "risk_score", *FEATURE_KEYS]


def sessions_to_frame(sessions: list) -> pd.DataFrame:
    """One row per session, oldest first. Missing features/risk become NaN."""
    rows = []
    for s in sessions:
        features = s.get("features") or {}
        risk = s.get("risk") or {}
        row = {
            "session_id": s.get("session_id"),
            "timestamp": s.get("timestamp"),
            "monitoring_status": s.get("monitoring_status"),
            "risk_score": risk.get("score"),
        }
        for key in FEATURE_KEYS:
            row[key] = features.get(key)
        rows.append(row)

    df = pd.DataFrame(rows, columns=SESSION_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    for column in ["risk_score", *FEATURE_KEYS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values("timestamp", na_position="last").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df


def compute_baseline(df: pd.DataFrame, feature: str, n: int = BASELINE_SESSIONS):
    """Mean and std of the first n sessions that have this feature, or None if fewer than n."""
    values = pd.to_numeric(df[feature], errors="coerce").dropna()
    if len(values) < n:
        return None
    first = values.iloc[:n]
    return {"mean": float(first.mean()), "std": float(first.std(ddof=0)), "n": n}


def latest_risk_score(df: pd.DataFrame):
    """risk.score of the most recent session that has one, else None."""
    scores = df["risk_score"].dropna()
    return float(scores.iloc[-1]) if len(scores) else None
