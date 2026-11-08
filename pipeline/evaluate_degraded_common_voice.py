
import json
from pathlib import Path

from backend.stt import transcribe_audio
from pipeline.evaluate_common_voice import calculate_wer


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

CONDITIONS = {
    "8khz": {
        "directory": PROJECT_ROOT / "data" / "degraded" / "8khz",
        "suffix": "_8khz.wav"
    },
    "noisy_10db": {
        "directory": PROJECT_ROOT / "data" / "degraded" / "noisy",
        "suffix": "_noisy_10dB.wav"
    }
}


def evaluate_condition(condition_name, condition, recordings):

    results_path = (
        RESULTS_DIR
        / f"common_voice_hi_27_{condition_name}_results.jsonl"
    )

    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_reference_words = 0

    successful = 0
    failed = 0
    processing_times = []

    print(f"\nEvaluating condition: {condition_name}")
    print("=" * 55)

    # Write results incrementally so completed recordings are preserved.
    with open(results_path, "w", encoding="utf-8") as results_file:

        for index, recording in enumerate(recordings, start=1):

            original_filename = Path(recording["filename"]).stem
            degraded_filename = (
                original_filename + condition["suffix"]
            )

            audio_path = (
                condition["directory"] / degraded_filename
            )

            print(
                f"\n[{index}/{len(recordings)}] "
                f"{degraded_filename}"
            )

            if not audio_path.is_file():

                print("Audio file missing.")

                result = {
                    "recording_id": recording["recording_id"],
                    "filename": degraded_filename,
                    "condition": condition_name,
                    "status": "failed",
                    "error": "Degraded audio file not found"
                }

                failed += 1

            else:

                try:
                    api_result = transcribe_audio(
                        audio_path=str(audio_path),
                        language_code="indic-auto"
                    )

                    if not api_result.get("success"):
                        raise RuntimeError(
                            "Gnani API reported unsuccessful transcription."
                        )

                    reference = recording["reference_transcript"]
                    hypothesis = api_result.get("transcript") or ""

                    metrics = calculate_wer(
                        reference,
                        hypothesis
                    )

                    result = {
                        "recording_id": recording["recording_id"],
                        "filename": degraded_filename,
                        "condition": condition_name,
                        "status": "success",
                        "detected_language": api_result.get(
                            "detected_language"
                        ),
                        "reference_transcript": reference,
                        "predicted_transcript": hypothesis,
                        **metrics,
                        "processing_time": api_result.get(
                            "processing_time"
                        ),
                        "request_id": api_result.get("request_id")
                    }

                    successful += 1

                    total_substitutions += metrics["substitutions"]
                    total_deletions += metrics["deletions"]
                    total_insertions += metrics["insertions"]
                    total_reference_words += metrics[
                        "reference_word_count"
                    ]

                    if metrics["wer"] is not None:
                        print(f"WER: {metrics['wer']:.2%}")

                    processing_time = api_result.get(
                        "processing_time"
                    )

                    if processing_time is not None:
                        processing_times.append(processing_time)

                except Exception as error:

                    print("Transcription failed:", error)

                    result = {
                        "recording_id": recording["recording_id"],
                        "filename": degraded_filename,
                        "condition": condition_name,
                        "status": "failed",
                        "error": str(error)
                    }

                    failed += 1

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
        "condition": condition_name,
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

    summary_path = (
        RESULTS_DIR
        / f"common_voice_hi_27_{condition_name}_summary.json"
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    print("\nCondition summary")
    print("-----------------")
    print("Successful:", successful)
    print("Failed:", failed)
    print("Reference words:", total_reference_words)
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

    print("Results:", results_path)
    print("Summary:", summary_path)

    return summary


def run_evaluation():

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        recordings = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("\nVoxSense - Degraded Audio Evaluation")
    print("====================================")
    print("Model: Gnani Prisma v3.0")
    print("Recordings:", len(recordings))
    print("Conditions:", len(CONDITIONS))
    print("Expected API requests:", len(recordings) * len(CONDITIONS))

    summaries = {}

    for condition_name, condition in CONDITIONS.items():

        summary = evaluate_condition(
            condition_name,
            condition,
            recordings
        )

        summaries[condition_name] = summary

    combined_summary_path = (
        RESULTS_DIR / "common_voice_hi_27_degraded_comparison.json"
    )

    with open(
        combined_summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summaries,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nDegraded Audio Evaluation Completed")
    print("===================================")

    for condition_name, summary in summaries.items():

        wer = summary["overall_wer"]

        if wer is not None:
            print(f"{condition_name}: {wer:.2%} WER")
        else:
            print(f"{condition_name}: WER unavailable")

    print("\nCombined comparison saved to:")
    print(combined_summary_path)


if __name__ == "__main__":
    run_evaluation()
