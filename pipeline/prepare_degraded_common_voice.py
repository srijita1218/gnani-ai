
import json
from pathlib import Path

from pipeline.degrade import degrade_audio


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "recordings"
    / "public"
    / "common_voice_hi_27"
)

METADATA_PATH = DATASET_DIR / "metadata.jsonl"

SNR_DB = 10


def prepare_degraded_dataset():

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        recordings = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("\nVoxSense - Common Voice Degraded Audio Preparation")
    print("==================================================")
    print("Recordings to process:", len(recordings))
    print("Noise level:", SNR_DB, "dB")

    successful = 0
    failed = 0

    for index, recording in enumerate(recordings, start=1):

        filename = recording["filename"]
        audio_path = DATASET_DIR / filename

        print(f"\n[{index}/{len(recordings)}] {filename}")

        try:
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Audio file not found: {audio_path}"
                )

            generated = degrade_audio(
                audio_path=audio_path,
                snr_db=SNR_DB
            )

            print("8 kHz:", generated["8khz"])
            print("Noisy:", generated["noisy"])

            successful += 1

        except Exception as error:
            print("Failed:", error)
            failed += 1

    print("\nPreparation Summary")
    print("===================")
    print("Successful recordings:", successful)
    print("Failed recordings:", failed)
    print("Expected degraded WAV files:", successful * 2)

    print("\nOutput directories:")
    print("8 kHz: data/degraded/8khz/")
    print("Noisy: data/degraded/noisy/")


if __name__ == "__main__":
    prepare_degraded_dataset()
