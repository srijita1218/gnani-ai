
import json
from pathlib import Path

from pipeline.run_pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECORDINGS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "recordings"
    / "self_recorded"
    / "converted"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "recordings.jsonl"
)

SPEAKER_ID = "self_001"


def get_processed_filenames():
    """Find recordings already present in the feature dataset."""

    processed = set()

    if DATASET_PATH.exists():
        with DATASET_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                if record.get("speaker_id") == SPEAKER_ID:
                    filename = record.get("metadata", {}).get("filename")
                    if filename:
                        processed.add(filename)

    return processed


def main():

    if not RECORDINGS_DIR.exists():
        raise FileNotFoundError(
            f"Recording directory not found: {RECORDINGS_DIR}"
        )

    wav_files = sorted(RECORDINGS_DIR.glob("*.wav"))
    processed = get_processed_filenames()

    print("\nVoxSense - Personal Recording Processing")
    print("========================================")
    print("WAV recordings found:", len(wav_files))
    print("Speaker ID:", SPEAKER_ID)

    successful = 0
    skipped = 0
    failed = 0

    for index, audio_path in enumerate(wav_files, start=1):

        print(f"\n[{index}/{len(wav_files)}] {audio_path.name}")

        if audio_path.name in processed:
            print("Already processed. Skipping.")
            skipped += 1
            continue

        try:
            recording = run_pipeline(
                audio_path=str(audio_path),
                speaker_id=SPEAKER_ID
            )

            print("Recording ID:", recording["recording_id"])
            successful += 1

        except Exception as error:
            print("Processing failed:", error)
            failed += 1

    print("\nProcessing Summary")
    print("==================")
    print("Successful:", successful)
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("Dataset:", DATASET_PATH)


if __name__ == "__main__":
    main()
