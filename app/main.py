"""FastAPI app: /health, /patients, /sessions."""
import json
import sqlite3
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .database import get_db, init_db
from .schemas import PatientIn, PatientOut, SessionIn


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="VoxSense API", version="0.1.0", lifespan=lifespan)

# allow the Vite dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


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
    if not db.execute(
        "SELECT 1 FROM patients WHERE id = ?", (body.participant_id,)
    ).fetchone():
        raise HTTPException(404, "Participant not found")

    consent = db.execute(
        "SELECT consent_given FROM consent WHERE patient_id = ? ORDER BY id DESC LIMIT 1",
        (body.participant_id,),
    ).fetchone()
    if consent is None or not consent["consent_given"]:
        raise HTTPException(403, "Participant has not given consent")

    payload = body.model_dump()
    session_id = payload.get("session_id") or str(uuid.uuid4())
    payload["session_id"] = session_id

    if db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
        raise HTTPException(409, f"Session {session_id} already exists")

    db.execute(
        "INSERT INTO sessions (id, patient_id, timestamp, payload) VALUES (?, ?, ?, ?)",
        (session_id, body.participant_id, body.timestamp, json.dumps(payload)),
    )
    db.commit()
    return payload


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
