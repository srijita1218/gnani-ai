
import json
import re
import unicodedata
from pathlib import Path

from backend.stt import transcribe_audio


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

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

RESULTS_PATH = RESULTS_DIR / "common_voice_hi_27_results.jsonl"
SUMMARY_PATH = RESULTS_DIR / "common_voice_hi_27_summary.json"


def normalize_text(text):
    """Normalize text for fair word-level comparison."""

    text = unicodedata.normalize("NFC", text or "")
    text = text.casefold()

    cleaned = []

    for character in text:
        if unicodedata.category(character).startswith("P"):
            cleaned.append(" ")
        else:
            cleaned.append(character)

    return re.sub(r"\s+", " ", "".join(cleaned)).strip()


def calculate_wer(reference, hypothesis):
    """Return word-level substitutions, deletions, insertions and WER."""

    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()

    rows = len(reference_words)
    columns = len(hypothesis_words)

    # Each cell stores (edit distance, operation).
    dp = [[(0, "") for _ in range(columns + 1)]
          for _ in range(rows + 1)]

    for i in range(1, rows + 1):
        dp[i][0] = (i, "D")

    for j in range(1, columns + 1):
        dp[0][j] = (j, "I")

    for i in range(1, rows + 1):
        for j in range(1, columns + 1):

            if reference_words[i - 1] == hypothesis_words[j - 1]:
                dp[i][j] = (dp[i - 1][j - 1][0], "M")
                continue

            substitution = dp[i - 1][j - 1][0] + 1
            deletion = dp[i - 1][j][0] + 1
            insertion = dp[i][j - 1][0] + 1

            best = min(substitution, deletion, insertion)

            if best == substitution:
                dp[i][j] = (best, "S")
            elif best == deletion:
                dp[i][j] = (best, "D")
            else:
                dp[i][j] = (best, "I")

    substitutions = 0
    deletions = 0
    insertions = 0

    i, j = rows, columns

    while i > 0 or j > 0:

        operation = dp[i][j][1]

        if operation == "M":
            i -= 1
            j -= 1

        elif operation == "S":
            substitutions += 1
            i -= 1
            j -= 1

        elif operation == "D":
            deletions += 1
            i -= 1

        elif operation == "I":
            insertions += 1
            j -= 1

    reference_word_count = len(reference_words)

    wer = (
        (substitutions + deletions + insertions) / reference_word_count
        if reference_word_count
        else None
    )

    return {
        "reference_word_count": reference_word_count,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": wer
    }


def run_evaluation():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        recordings = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("\nVoxSense - Common Voice Hindi Evaluation")
    print("========================================")
    print("Model: Gnani Prisma v3.0")
    print("Recordings:", len(recordings))

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_reference_words = 0

    successful = 0
    failed = 0
    processing_times = []

    # Start a fresh results file for this evaluation run.
    with open(RESULTS_PATH, "w", encoding="utf-8") as results_file:

        for index, recording in enumerate(recordings, start=1):

            filename = recording["filename"]
            audio_path = DATASET_DIR / filename

            print(f"\n[{index}/{len(recordings)}] Processing {filename}")

            if not audio_path.is_file():
                print("Audio file missing. Skipping.")

                result = {
                    "recording_id": recording["recording_id"],
                    "filename": filename,
                    "status": "failed",
                    "error": "Audio file not found"
                }

                failed += 1

            else:
                try:
                    api_result = transcribe_audio(
                        audio_path=str(audio_path),
                        language_code="indic-auto"
                    )

                    if not api_result.get("success"):
                        raise RuntimeError("Gnani API reported unsuccessful transcription.")

                    hypothesis = api_result.get("transcript") or ""
                    reference = recording["reference_transcript"]

                    metrics = calculate_wer(reference, hypothesis)

                    result = {
                        "recording_id": recording["recording_id"],
                        "filename": filename,
                        "status": "success",
                        "detected_language": api_result.get("detected_language"),
                        "reference_transcript": reference,
                        "predicted_transcript": hypothesis,
                        **metrics,
                        "processing_time": api_result.get("processing_time"),
                        "request_id": api_result.get("request_id")
                    }

                    successful += 1

                    total_substitutions += metrics["substitutions"]
                    total_deletions += metrics["deletions"]
                    total_insertions += metrics["insertions"]
                    total_reference_words += metrics["reference_word_count"]

                    if metrics["wer"] is not None:
                        print(f"WER: {metrics['wer']:.2%}")

                    if api_result.get("processing_time") is not None:
                        processing_times.append(
                            api_result["processing_time"]
                        )

                except Exception as error:

                    print("Transcription failed:", error)

                    result = {
                        "recording_id": recording["recording_id"],
                        "filename": filename,
                        "status": "failed",
                        "error": str(error)
                    }

                    failed += 1

            # Save after every clip so completed results are retained.
            results_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            results_file.flush()

    total_errors = (
        total_substitutions
        + total_deletions
        + total_insertions
    )

    overall_wer = (
        total_errors / total_reference_words
        if total_reference_words
        else None
    )

    summary = {
        "dataset": "Mozilla Common Voice Scripted Speech 27.0",
        "locale": "hi",
        "model": "prisma-v3.0",
        "total_recordings": len(recordings),
        "successful_recordings": successful,
        "failed_recordings": failed,
        "reference_word_count": total_reference_words,
        "substitutions": total_substitutions,
        "deletions": total_deletions,
        "insertions": total_insertions,
        "total_word_errors": total_errors,
        "overall_wer": overall_wer,
        "average_api_processing_time": (
            sum(processing_times) / len(processing_times)
            if processing_times else None
        )
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    print("\nEvaluation completed.")
    print("=====================")
    print("Successful:", successful)
    print("Failed:", failed)
    print("Total reference words:", total_reference_words)
    print("Substitutions:", total_substitutions)
    print("Deletions:", total_deletions)
    print("Insertions:", total_insertions)

    if overall_wer is not None:
        print(f"Overall WER: {overall_wer:.2%}")
    else:
        print("Overall WER: unavailable")

    if processing_times:
        print(
            "Average API processing time:",
            round(summary["average_api_processing_time"], 4),
            "seconds"
        )

    print("\nResults saved to:", RESULTS_PATH)
    print("Summary saved to:", SUMMARY_PATH)


if __name__ == "__main__":
    run_evaluation()
