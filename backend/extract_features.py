
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import librosa
import numpy as np
import soundfile as sf


DATASET_PATH = Path("data/datasets/recordings.jsonl")
TARGET_SAMPLE_RATE = 16000


def extract_features(
    audio_path,
    speaker_id,
    recorded_at=None,
    transcription=None
):
    """
    Extract acoustic features and metadata from an audio recording.

    Returns a structured recording record and appends it
    to the VoxSense JSONL dataset.
    """

    audio_path = Path(audio_path)

    # 1. Validate input
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise ValueError("Audio path must point to a file.")

    if not speaker_id or not speaker_id.strip():
        raise ValueError("speaker_id cannot be empty.")

    if audio_path.stat().st_size == 0:
        raise ValueError("Audio file is empty.")

    # 2. Read original audio metadata
    audio_info = sf.info(str(audio_path))

    original_sample_rate = audio_info.samplerate
    channels = audio_info.channels
    file_size = audio_path.stat().st_size
    original_duration = audio_info.duration

    # 3. Basic quality validation
    if original_duration < 1.0:
        raise ValueError("Recording is too short. Minimum duration is 1 second.")

    # 4. Load audio for acoustic analysis
    audio, sample_rate = librosa.load(
        str(audio_path),
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    if len(audio) == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid or empty signal data.")

    # 5. Calculate RMS energy
    rms = librosa.feature.rms(y=audio)[0]
    average_energy = float(np.mean(rms))

    if average_energy < 1e-5:
        raise ValueError("Recording appears to contain silence.")

    # 6. Estimate pitch
    pitch, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=65,
        fmax=400,
        sr=sample_rate
    )

    valid_pitch = pitch[~np.isnan(pitch)]

    average_pitch = (
        float(np.mean(valid_pitch))
        if len(valid_pitch) > 0 else None
    )

    pitch_variability = (
        float(np.std(valid_pitch))
        if len(valid_pitch) > 0 else None
    )

    # 7. Detect speech regions
    intervals = librosa.effects.split(
        audio,
        top_db=30
    )

    speech_duration = sum(
        (end - start) / sample_rate
        for start, end in intervals
    )

    # 8. Estimate pauses
    pause_duration = max(
        0.0,
        duration - speech_duration
    )

    pause_count = max(
        0,
        len(intervals) - 1
    )

    # 9. Audio quality indicators
    clipping_ratio = float(
        np.mean(np.abs(audio) >= 0.99)
    )

    quality_flags = []

    if clipping_ratio > 0.01:
        quality_flags.append("possible_audio_clipping")

    if speech_duration < 0.5:
        quality_flags.append("very_little_detected_speech")

    # 10. Generate recording identifiers and timestamps
    recording_id = str(uuid.uuid4())

    processed_at = datetime.now(
        timezone.utc
    ).isoformat()

    # 11. Build structured recording record
    recording = {
        "recording_id": recording_id,
        "speaker_id": speaker_id,
        "transcription": transcription,

        "metadata": {
            "filename": audio_path.name,
            "file_size_bytes": file_size,
            "duration_seconds": round(original_duration, 2),
            "original_sample_rate": original_sample_rate,
            "analysis_sample_rate": sample_rate,
            "channels": channels,
            "recorded_at": recorded_at,
            "processed_at": processed_at
        },

        "features": {
            "average_pitch_hz": (
                round(average_pitch, 2)
                if average_pitch is not None else None
            ),

            "pitch_variability_hz": (
                round(pitch_variability, 2)
                if pitch_variability is not None else None
            ),

            "average_rms_energy": round(average_energy, 4),

            "speech_duration_seconds": round(
                speech_duration, 2
            ),

            "pause_duration_seconds": round(
                pause_duration, 2
            ),

            "pause_count": pause_count,

            "clipping_ratio": round(clipping_ratio, 4)
        },

        "quality": {
            "valid": True,
            "flags": quality_flags
        }
    }

    # 12. Save recording to JSONL dataset
    DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(DATASET_PATH, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(recording, ensure_ascii=False) + "\n"
        )

    print("\nVoxSense - Feature Extraction")
    print("============================")
    print("Recording ID:", recording_id)
    print("Speaker ID:", speaker_id)
    print("Duration:", duration)
    print("Dataset:", DATASET_PATH)
    print("\nExtracted Features:")
    print(json.dumps(recording, indent=4, ensure_ascii=False))

    return recording


if __name__ == "__main__":
    extract_features(
        audio_path="data/sample.wav",
        speaker_id="user_001"
    )
