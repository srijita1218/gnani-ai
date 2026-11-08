
import argparse
from pathlib import Path

from backend.stt import transcribe_audio


def run_stt(audio_path, language_code="indic-auto"):

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if not audio_path.is_file():
        raise ValueError(
            f"Expected an audio file: {audio_path}"
        )

    print("\nVoxSense - Gnani STT")
    print("===================")
    print("Audio:", audio_path.name)
    print("Model: prisma-v3.0")

    result = transcribe_audio(
        audio_path=str(audio_path),
        language_code=language_code
    )

    print("\nTranscription completed.")
    print("Detected language:", result["detected_language"])
    print("Truncated:", result["truncated"])

    print("\nTranscript:")
    print(result["transcript"])

    print("\nTimestamp information:")
    print("Timestamp type:", result["timestamp_type"])
    print("Timestamps:", result["timestamps"])

    print("\nAPI information:")
    print("Request ID:", result["request_id"])
    print("Processing time:", result["processing_time"])
    print("End-to-end latency:", result["end_to_end_latency"])

    return result


def main():

    parser = argparse.ArgumentParser(
        description="Transcribe audio using Gnani Prisma v3."
    )

    parser.add_argument(
        "audio_path",
        help="Path to the input audio file."
    )

    parser.add_argument(
        "--language",
        default="indic-auto",
        help="Language code. Defaults to indic-auto."
    )

    args = parser.parse_args()

    try:
        run_stt(
            audio_path=args.audio_path,
            language_code=args.language
        )

    except (FileNotFoundError, ValueError) as error:
        print("\nError:", error)


if __name__ == "__main__":
    main()
