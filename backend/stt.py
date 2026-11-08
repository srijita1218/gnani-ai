
import os
from pathlib import Path

from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def transcribe_audio(audio_path, language_code="indic-auto"):
    """
    Transcribe an audio file using Gnani Prisma v3.

    Returns the transcript, detected language, API metadata,
    and timestamp availability.
    """

    api_key = os.getenv("GNANI_API_KEY")

    if not api_key:
        raise ValueError("GNANI_API_KEY is missing from .env")

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise ValueError(f"Expected an audio file: {audio_path}")

    client = GnaniSTTClient(api_key=api_key)

    result = client.transcribe(
        str(audio_path),
        language_code=language_code,
        model="prisma-v3.0",
        format="verbatim"
    )

    # The inspected REST response does not contain word-level timestamps.
    timestamps = result.get("timestamps")

    return {
        "transcript": result.get("transcript"),
        "detected_language": result.get("detected_language_code"),
        "truncated": result.get("truncated"),
        "timestamps": timestamps,
        "timestamp_type": (
            "word_level"
            if timestamps is not None
            else "not_available_from_rest"
        ),
        "model": result.get("model"),
        "request_id": result.get("request_id"),
        "success": result.get("success"),
        "processing_time": result.get("processing_time"),
        "end_to_end_latency": result.get("end_to_end_latency")
    }
