
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "recordings"
    / "self_recorded"
)

OUTPUT_DIR = INPUT_DIR / "converted"


def convert_recordings():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Recording folder not found: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    recordings = sorted(INPUT_DIR.glob("*.mp4"))

    print("\nVoxSense - MP4 to WAV Conversion")
    print("================================")
    print("MP4 recordings found:", len(recordings))

    if not recordings:
        print("No MP4 files found.")
        return

    successful = 0
    failed = 0

    for index, mp4_path in enumerate(recordings, start=1):

        wav_path = OUTPUT_DIR / f"{mp4_path.stem}.wav"

        print(f"\n[{index}/{len(recordings)}] {mp4_path.name}")

        command = [
            "ffmpeg",
            "-y",
            "-i", str(mp4_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(wav_path)
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )

            print("Converted:", wav_path.name)
            successful += 1

        except subprocess.CalledProcessError as error:
            print("Conversion failed:")
            print(error.stderr[-1500:])
            failed += 1

    print("\nConversion Summary")
    print("==================")
    print("Successful:", successful)
    print("Failed:", failed)
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_recordings()
