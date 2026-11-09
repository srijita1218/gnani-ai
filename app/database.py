"""SQLite setup: connection helper, schema, FastAPI dependency."""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id            TEXT PRIMARY KEY,       -- the participant_id, e.g. 'P001'
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
    id            TEXT PRIMARY KEY,       -- the session_id, e.g. 'SESSION_001'
    patient_id    TEXT NOT NULL REFERENCES patients(id),
    timestamp     TEXT,                   -- from the session JSON
    payload       TEXT NOT NULL,          -- full session JSON as sent
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);
"""


def db_path() -> str:
    # read at call time so tests can point DB_PATH at a temp file
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
