"""Pydantic request/response models."""
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PatientIn(BaseModel):
    participant_id: Optional[str] = None   # e.g. "P001"; generated if omitted
    display_name: Optional[str] = None     # pseudonym only, never a real name
    consent_given: bool
    consent_text: Optional[str] = (
        "I agree to my voice recordings being used for this research prototype."
    )


class PatientOut(BaseModel):
    id: str                                # same value as participant_id
    participant_id: str
    display_name: Optional[str] = None
    consent_given: bool
    created_at: str


class SessionIn(BaseModel):
    """Only participant_id is required. All other fields in mock_session.json
    (recording, transcription, features, baseline, ...) are accepted and stored
    as-is, so the contract can evolve without database migrations."""
    model_config = ConfigDict(extra="allow")

    participant_id: str
    session_id: Optional[str] = None       # generated if omitted
    timestamp: Optional[str] = None
