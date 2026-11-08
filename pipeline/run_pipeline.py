
from pathlib import Path

from backend.stt import transcribe_audio
from backend.extract_features import extract_features


def run_pipeline(
    audio_path,
    speaker_id,
    language_code="indic-auto",
    recorded_at=None
):
    """
    Run the complete VoxSense processing pipeline.

    1. Transcribe audio using Gnani Prisma v3.
    2. Extract acoustic features using Librosa.
    3. Save everything in one recording entry.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("\nVoxSense - Complete Processing Pipeline")
    print("=======================================")

    # Step 1: Gnani speech-to-text
    print("\n[1/2] Running Gnani Prisma v3 transcription...")

    transcription = transcribe_audio(
        audio_path=str(audio_path),
        language_code=language_code
    )

    print("Transcription completed.")
    print("Detected language:", transcription["detected_language"])

    # Step 2: Acoustic feature extraction
    print("\n[2/2] Extracting acoustic features...")

    recording = extract_features(
        audio_path=str(audio_path),
        speaker_id=speaker_id,
        recorded_at=recorded_at,
        transcription=transcription
    )

    print("\nPipeline completed successfully.")
    print("Recording ID:", recording["recording_id"])

    return recording


if __name__ == "__main__":
    run_pipeline(
        audio_path="data/sample.wav",
        speaker_id="user_001"
    )
