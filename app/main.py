"""FastAPI app: /health, /patients, /sessions, /sessions/audio."""
import json
import sqlite3
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .audio_pipeline import build_session_from_audio
from .database import get_db, init_db
from .referrals import create_referral_if_high_risk, get_referral_queue
from .schemas import PatientIn, PatientOut, SessionIn

MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="VoxSense API", version="0.2.0", lifespan=lifespan)

# allow the Vite dev server and the Streamlit UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8501", "http://127.0.0.1:8501",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- helpers --

def _require_consenting_patient(db: sqlite3.Connection, participant_id: str) -> None:
    if not db.execute("SELECT 1 FROM patients WHERE id = ?", (participant_id,)).fetchone():
        raise HTTPException(404, "Participant not found")
    consent = db.execute(
        "SELECT consent_given FROM consent WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
        (participant_id,),
    ).fetchone()
    if consent is None or not consent["consent_given"]:
        raise HTTPException(403, "Participant has not given consent")


def _store_session(db: sqlite3.Connection, payload: dict) -> dict:
    session_id = payload.get("session_id") or str(uuid.uuid4())
    payload["session_id"] = session_id

    if db.execute(
        "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
    ).fetchone():
        raise HTTPException(409, f"Session {session_id} already exists")

    db.execute(
        "INSERT INTO sessions (id, patient_id, timestamp, payload) VALUES (?, ?, ?, ?)",
        (
            session_id,
            payload["participant_id"],
            payload.get("timestamp"),
            json.dumps(payload),
        ),
    )

    create_referral_if_high_risk(db, payload)
    db.commit()

    return payload


# ----------------------------------------------------------------- routes --

@app.get("/")
def root():
    return {"message": "VoxSense API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/referrals")
def list_referrals(db: sqlite3.Connection = Depends(get_db)):
    """Doctor's queue of high-risk session referrals."""
    return get_referral_queue(db)


@app.post("/patients", response_model=PatientOut, status_code=201)
def create_patient(body: PatientIn, db: sqlite3.Connection = Depends(get_db)):
    pid = body.participant_id or f"P{uuid.uuid4().hex[:8].upper()}"
    if db.execute("SELECT 1 FROM patients WHERE id = ?", (pid,)).fetchone():
        raise HTTPException(409, f"Participant {pid} already exists")

    db.execute(
        "INSERT INTO patients (id, display_name) VALUES (?, ?)",
        (pid, body.display_name),
    )
    db.execute(
        "INSERT INTO consent (patient_id, consent_given, consent_text) VALUES (?, ?, ?)",
        (pid, int(body.consent_given), body.consent_text),
    )
    db.commit()
    row = db.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
    return PatientOut(
        id=row["id"],
        participant_id=row["id"],
        display_name=row["display_name"],
        consent_given=body.consent_given,
        created_at=row["created_at"],
    )


@app.post("/sessions", status_code=201)
def create_session(body: SessionIn, db: sqlite3.Connection = Depends(get_db)):
    _require_consenting_patient(db, body.participant_id)
    return _store_session(db, body.model_dump())


@app.post("/sessions/audio", status_code=201)
async def create_session_from_audio(
    request: Request,
    participant_id: str,
    language: str = "indic-auto",
    filename: str = "recording.wav",
    db: sqlite3.Connection = Depends(get_db),
):
    """Raw audio bytes in the request body (no multipart needed).

    Extracts features, transcribes if possible, and saves a session. The audio
    itself is processed from a temporary file that is deleted afterwards."""
    _require_consenting_patient(db, participant_id)

    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(415, f"Unsupported audio type '{suffix}'. Use .wav, .mp3 or .flac")

    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(400, "Request body is empty")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio file is larger than 25 MB")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(audio_bytes)
        try:
            payload = await run_in_threadpool(
                build_session_from_audio, path, participant_id, language, safe_name
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))

    return _store_session(db, payload)


@app.get("/patients/{patient_id}/sessions")
def list_sessions(patient_id: str, db: sqlite3.Connection = Depends(get_db)):
    if not db.execute("SELECT 1 FROM patients WHERE id = ?", (patient_id,)).fetchone():
        raise HTTPException(404, "Participant not found")

    rows = db.execute(
        "SELECT payload FROM sessions WHERE patient_id = ? "
        "ORDER BY COALESCE(timestamp, created_at) ASC",
        (patient_id,),
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows]
