"""Turn an uploaded audio file into a contract session dict.

Heavy dependencies (librosa, parselmouth, the Gnani SDK) are imported inside the
functions, so the API still starts and M2 still works when they are not installed.

Raises:
    ValueError    - the audio is unusable (too short, corrupt): the API returns 422
    RuntimeError  - a required dependency is missing: the API returns 503
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.1"
BASELINE_NOTE = "Additional valid recordings are required to establish a personal baseline."


def _transcribe(audio_path: Path, language: str) -> dict:
    """Gnani Prisma transcription. Never raises: on any failure the session is
    still saved with its acoustic features and status 'unavailable'."""
    block = {
        "model": "prisma-v3.0",
        "language": None,
        "text": None,
        "status": "unavailable",
        "timestamps": None,
        "timestamp_type": "not_available_from_rest",
    }
    try:
        from backend.stt import transcribe_audio

        result = transcribe_audio(audio_path=str(audio_path), language_code=language)
    except Exception as exc:  # missing SDK/key, network, API error
        log.warning("Transcription unavailable: %s: %s", type(exc).__name__, exc)
        block["error"] = type(exc).__name__   # type only: messages could echo secrets
        return block

    block.update(
        language=result.get("detected_language"),
        text=result.get("transcript"),
        status="completed",
        timestamps=result.get("timestamps"),
        timestamp_type=result.get("timestamp_type", "not_available_from_rest"),
    )
    return block


def _audio_info(audio_path: Path) -> dict:
    info = {"sample_rate": None, "channels": None, "duration_seconds": None}
    try:
        import soundfile as sf

        meta = sf.info(str(audio_path))
        info.update(
            sample_rate=meta.samplerate,
            channels=meta.channels,
            duration_seconds=round(meta.duration, 2),
        )
    except Exception:  # e.g. mp3 on an older libsndfile: leave as None
        pass
    return info


def build_session_from_audio(
    audio_path: Path, participant_id: str, language: str, original_filename: str
) -> dict:
    try:
        from pipeline.features import extract
    except ImportError as exc:
        raise RuntimeError(
            "Feature extraction dependencies are missing. "
            "Run: pip install -r requirements-m3.txt"
        ) from exc

    # Validate the audio first (ValueError if too short or corrupt) so we do not
    # spend a speech-to-text API call on an unusable file.
    features = extract(audio_path)

    transcription = _transcribe(audio_path, language)
    if transcription["text"]:
        features = extract(audio_path, transcript=transcription["text"])  # adds speaking_rate

    now = datetime.now(timezone.utc).astimezone()
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": f"SESSION_{uuid.uuid4().hex[:8].upper()}",
        "assessment_id": f"ASSESSMENT_{uuid.uuid4().hex[:8].upper()}",
        "participant_id": participant_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "recording": {
            "recording_id": f"REC_{uuid.uuid4().hex[:8].upper()}",
            "filename": original_filename,
            **_audio_info(audio_path),
            "audio_quality": None,
            "source": "ui_upload",
        },
        "transcription": transcription,
        "features": features,
        "baseline": {"available": False, "recordings_used": 0},
        "comparison": {
            "speaking_rate_deviation": None,
            "pause_duration_deviation": None,
            "pitch_variation_deviation": None,
            "rms_energy_deviation": None,
        },
        "monitoring_status": "insufficient_baseline",
        "recommendation": BASELINE_NOTE,
        "metadata": {"data_type": "user_recording", "is_clinically_validated": False},
    }
