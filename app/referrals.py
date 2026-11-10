"""Referral creation and retrieval."""

import math
import sqlite3

from .notes import generate_notes

REFERRAL_THRESHOLD = 0.66


def create_referral_if_high_risk(
    db: sqlite3.Connection,
    session: dict,
) -> bool:
    """Create one referral when a session has a valid high-risk score."""

    risk = session.get("risk") or {}
    if not isinstance(risk, dict):
        return False

    score = risk.get("score")

    if isinstance(score, bool):
        return False

    try:
        score = float(score)
    except (TypeError, ValueError):
        return False

    if not math.isfinite(score) or not 0 <= score <= 1:
        return False

    if score < REFERRAL_THRESHOLD:
        return False

    session_id = session.get("session_id")
    patient_id = session.get("participant_id")

    if not session_id or not patient_id:
        return False

    # Avoid duplicate referrals for the same session.
    existing = db.execute(
        "SELECT 1 FROM referrals WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    if existing:
        return False

    domain = str(risk.get("domain") or "Speech monitoring")
    transcription = session.get("transcription") or {}
    language = transcription.get("language") or "en-IN"

    notes = generate_notes(
        patient_id=patient_id,
        score=score,
        domain=domain,
        patient_language=language,
    )

    db.execute(
        """
        INSERT INTO referrals (
            session_id, patient_id, domain, score, status,
            clinician_note, patient_message, patient_language
        )
        VALUES (?, ?, ?, ?, 'Pending', ?, ?, ?)
        """,
        (
            session_id,
            patient_id,
            domain,
            score,
            notes["clinician_note"],
            notes["patient_message"],
            notes["patient_language"],
        ),
    )

    return True


def get_referral_queue(db: sqlite3.Connection) -> list[dict]:
    """Return referrals in newest-first order."""

    rows = db.execute(
        """
        SELECT
            r.id,
            r.session_id,
            r.patient_id,
            COALESCE(p.display_name, r.patient_id) AS patient,
            r.domain,
            r.score,
            r.created_at,
            r.status,
            r.clinician_note,
            r.patient_message,
            r.patient_language
        FROM referrals r
        LEFT JOIN patients p ON p.id = r.patient_id
        ORDER BY r.created_at DESC, r.id DESC
        """
    ).fetchall()

    return [dict(row) for row in rows]
