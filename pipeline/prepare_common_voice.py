
import csv
import json
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = Path(
    r"C:\Users\SRIJITA\Downloads\1789492443485-cv-corpus-27.0-2026-09-11-hi"
    r"\cv-corpus-27.0-2026-09-11\hi"
)

CLIPS_DIR = DATASET_DIR / "clips"
TSV_PATH = DATASET_DIR / "validated.tsv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "recordings"
    / "public"
    / "common_voice_hi_27"
)

METADATA_PATH = OUTPUT_DIR / "metadata.jsonl"

NUMBER_OF_CLIPS = 10
RANDOM_SEED = 42


def prepare_dataset():

    if not TSV_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found: {TSV_PATH}")

    if not CLIPS_DIR.exists():
        raise FileNotFoundError(f"Audio clips folder not found: {CLIPS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nVoxSense - Common Voice Dataset Preparation")
    print("============================================")

    # Read the TSV using UTF-8 to preserve Hindi characters.
    with open(TSV_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        rows = list(reader)

    print("Validated metadata rows:", len(rows))

    # Randomize selection while keeping the result reproducible.
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(rows)

    selected = []
    used_speakers = set()

    for row in rows:

        speaker_id = row.get("client_id", "").strip()
        filename = row.get("path", "").strip()
        transcript = row.get("sentence", "").strip()

        if not speaker_id or not filename or not transcript:
            continue

        # Avoid selecting multiple clips from the same speaker.
        if speaker_id in used_speakers:
            continue

        # Only accept filenames, not arbitrary paths.
        if Path(filename).name != filename:
            continue

        source_audio = CLIPS_DIR / filename

        if not source_audio.is_file():
            continue

        selected.append({
            "speaker_id": speaker_id,
            "filename": filename,
            "transcript": transcript,
            "source_audio": source_audio
        })

        used_speakers.add(speaker_id)

        if len(selected) == NUMBER_OF_CLIPS:
            break

    if not selected:
        raise RuntimeError("No usable recordings were found.")

    if len(selected) < NUMBER_OF_CLIPS:
        print("Warning: Only found", len(selected), "usable recordings.")

    metadata_records = []

    for index, item in enumerate(selected, start=1):

        recording_filename = f"common_voice_hi_{index:02d}.mp3"
        destination = OUTPUT_DIR / recording_filename

        shutil.copy2(item["source_audio"], destination)

        metadata_records.append({
            "recording_id": f"CVHI{index:03d}",
            "filename": recording_filename,
            "source": "Mozilla Common Voice",
            "dataset_version": "27.0",
            "locale": "hi",
            "speaker_id": item["speaker_id"],
            "reference_transcript": item["transcript"],
            "audio_format": "mp3"
        })

        print(f"Copied {index}/{len(selected)}: {recording_filename}")

    # Save metadata as UTF-8 JSONL.
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        for record in metadata_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nDataset preparation completed.")
    print("Recordings copied:", len(metadata_records))
    print("Audio directory:", OUTPUT_DIR)
    print("Metadata file:", METADATA_PATH)


if __name__ == "__main__":
    prepare_dataset()
