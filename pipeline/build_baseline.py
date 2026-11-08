
import hashlib
import json
import statistics
from pathlib import Path


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline.json"
)

SPEAKER_ID = "self_001"

FEATURE_NAMES = [
    "average_pitch_hz",
    "pitch_variability_hz",
    "average_rms_energy",
    "speech_duration_seconds",
    "pause_duration_seconds",
    "pause_count",
    "clipping_ratio"
]


def calculate_sha256(file_path):
    """Calculate a file's SHA-256 hash."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_personal_records():
    """Load feature records belonging to the selected speaker."""

    records = {}

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record.get("speaker_id") != SPEAKER_ID:
                continue

            filename = record.get("metadata", {}).get("filename")

            if filename:
                records[filename] = record

    return records


def build_baseline():

    if not RECORDINGS_DIR.exists():
        raise FileNotFoundError(
            f"Recording directory not found: {RECORDINGS_DIR}"
        )

    records = load_personal_records()
    wav_files = sorted(RECORDINGS_DIR.glob("*.wav"))

    unique_records = []
    seen_hashes = {}
    duplicates = []

    print("\nVoxSense - Personal Baseline Builder")
    print("====================================")

    for wav_path in wav_files:

        filename = wav_path.name

        if filename not in records:
            print("Skipping: no feature record for", filename)
            continue

        record = records[filename]

        if not record.get("quality", {}).get("valid", False):
            print("Skipping invalid recording:", filename)
            continue

        file_hash = calculate_sha256(wav_path)

        if file_hash in seen_hashes:
            duplicates.append({
                "filename": filename,
                "duplicate_of": seen_hashes[file_hash]
            })

            print(
                "Duplicate excluded:",
                filename,
                "matches",
                seen_hashes[file_hash]
            )
            continue

        seen_hashes[file_hash] = filename

        unique_records.append(record)

        print("Included:", filename)

    if len(unique_records) < 2:
        raise ValueError(
            "At least two unique valid recordings are required."
        )

    feature_summary = {}

    for feature_name in FEATURE_NAMES:

        values = [
            record["features"][feature_name]
            for record in unique_records
            if record.get("features", {}).get(feature_name) is not None
        ]

        if not values:
            continue

        feature_summary[feature_name] = {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "std_dev": round(statistics.pstdev(values), 4),
            "minimum": round(min(values), 4),
            "maximum": round(max(values), 4)
        }

    baseline = {
        "speaker_id": SPEAKER_ID,
        "baseline_type": "initial_descriptive_baseline",
        "unique_recording_count": len(unique_records),
        "duplicate_recordings_excluded": duplicates,
        "included_recordings": [
            record["metadata"]["filename"]
            for record in unique_records
        ],
        "features": feature_summary,
        "limitations": [
            "This is a small preliminary baseline.",
            "Recordings are short and may not represent normal speech over time.",
            "Statistics are descriptive and are not diagnostic thresholds.",
            "Repeated sessions under consistent recording conditions are needed."
        ]
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            baseline,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nBaseline Summary")
    print("================")
    print("Unique recordings:", len(unique_records))
    print("Duplicates excluded:", len(duplicates))
    print("Output:", OUTPUT_PATH)

    print("\nFeature statistics:")

    for feature_name, values in feature_summary.items():
        print(
            feature_name,
            "| Mean:", values["mean"],
            "| Median:", values["median"]
        )

    return baseline


if __name__ == "__main__":
    build_baseline()
