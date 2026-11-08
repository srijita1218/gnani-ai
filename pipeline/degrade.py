
import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def create_8khz_copy(audio, original_sample_rate):
    """Resample audio to 8 kHz."""

    return librosa.resample(
        audio,
        orig_sr=original_sample_rate,
        target_sr=8000,
        axis=0
    )


def create_noisy_copy(audio, snr_db=10, seed=42):
    """
    Add white Gaussian noise at the requested SNR.

    Higher SNR means less noise.
    Lower SNR means more noise.
    """

    signal_rms = np.sqrt(np.mean(audio ** 2))

    if signal_rms < 1e-8:
        raise ValueError("Cannot add noise to a silent recording.")

    rng = np.random.default_rng(seed)

    noise = rng.standard_normal(audio.shape)

    noise_rms = np.sqrt(np.mean(noise ** 2))

    desired_noise_rms = signal_rms / (10 ** (snr_db / 20))

    noise = noise * (desired_noise_rms / noise_rms)

    noisy_audio = audio + noise

    # Prevent clipping when writing a PCM WAV file.
    peak = np.max(np.abs(noisy_audio))

    if peak > 0.99:
        noisy_audio = noisy_audio * (0.99 / peak)

    return noisy_audio


def degrade_audio(audio_path, snr_db=10):
    """Generate 8 kHz and noisy copies of an audio file."""

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not audio_path.is_file():
        raise ValueError(f"Expected an audio file: {audio_path}")

    if audio_path.stat().st_size == 0:
        raise ValueError("Audio file is empty.")

    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True
    )

    if len(audio) == 0:
        raise ValueError("Audio contains no samples.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numerical values.")

    output_root = Path("data/degraded")

    eight_khz_dir = output_root / "8khz"
    noisy_dir = output_root / "noisy"

    eight_khz_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    filename = audio_path.stem

    # Generate 8 kHz version.
    audio_8khz = create_8khz_copy(audio, sample_rate)

    eight_khz_path = eight_khz_dir / f"{filename}_8khz.wav"

    sf.write(
        str(eight_khz_path),
        audio_8khz,
        8000,
        subtype="PCM_16"
    )

    # Generate noisy version at the original sample rate.
    noisy_audio = create_noisy_copy(audio, snr_db=snr_db)

    noisy_path = noisy_dir / f"{filename}_noisy_{snr_db}dB.wav"

    sf.write(
        str(noisy_path),
        noisy_audio,
        sample_rate,
        subtype="PCM_16"
    )

    print("\nVoxSense - Audio Degradation")
    print("============================")
    print("Original:", audio_path)
    print("Original sample rate:", sample_rate)
    print("Original duration:", round(len(audio) / sample_rate, 2), "seconds")

    print("\nGenerated files:")
    print("8 kHz copy:", eight_khz_path)
    print("Noisy copy:", noisy_path)
    print("Noise SNR:", snr_db, "dB")

    return {
        "original": str(audio_path),
        "8khz": str(eight_khz_path),
        "noisy": str(noisy_path),
        "snr_db": snr_db
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate degraded audio copies for VoxSense evaluation."
    )

    parser.add_argument(
        "audio_path",
        help="Path to the original WAV audio."
    )

    parser.add_argument(
        "--snr",
        type=float,
        default=10,
        help="Noise signal-to-noise ratio in dB. Defaults to 10."
    )

    args = parser.parse_args()

    try:
        degrade_audio(
            audio_path=args.audio_path,
            snr_db=args.snr
        )

    except (FileNotFoundError, ValueError) as error:
        print("\nError:", error)


if __name__ == "__main__":
    main()
