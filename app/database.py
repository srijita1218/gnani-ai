
"""SQLite setup: connection helper, schema, FastAPI dependency."""

import os
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id            TEXT PRIMARY KEY,
    display_name  TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS consent (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    TEXT NOT NULL REFERENCES patients(id),
    consent_given INTEGER NOT NULL,
    consent_text  TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    patient_id    TEXT NOT NULL REFERENCES patients(id),
    timestamp     TEXT,
    payload       TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_patient
ON sessions(patient_id);

CREATE TABLE IF NOT EXISTS referrals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL UNIQUE REFERENCES sessions(id),
    patient_id        TEXT NOT NULL REFERENCES patients(id),
    domain            TEXT NOT NULL,
    score             REAL NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    status            TEXT NOT NULL DEFAULT 'Pending',
    clinician_note    TEXT NOT NULL,
    patient_message   TEXT NOT NULL,
    patient_language  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_referrals_created
ON referrals(created_at);
"""


def db_path() -> str:
    # Read at call time so tests can point DB_PATH at a temporary file.
    return os.environ.get("DB_PATH", "data/app.db")


def get_conn() -> sqlite3.Connection:
    path = db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with closing(get_conn()) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def get_db():
    """FastAPI dependency: one connection per request."""
    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()
