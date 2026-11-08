
from backend.stt import transcribe_audio

audio_path = "data/sample.wav"


def main():
    result = transcribe_audio(audio_path)

    print("\nVoxSense - Gnani STT Results")
    print("============================")
    print("Transcript:", result["transcript"])
    print("Detected language:", result["detected_language"])
    print("Truncated:", result["truncated"])


if __name__ == "__main__":
    main()
