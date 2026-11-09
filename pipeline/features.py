"""M3: feature extraction.

extract(wav_path, transcript=None) -> dict

The returned dict is the `features` block of the session contract
(contract/mock_session.json). It always contains exactly FEATURE_KEYS.
A value is a finite float, or None when it cannot be measured (for example
pitch in a silent clip). It is plain JSON: json.dumps(features) always works.

Timing/energy features use numpy only. Voice-quality features use Praat via
parselmouth. Audio is loaded with librosa at 16 kHz mono, so 8 kHz and noisy
copies from degrade.py work without special handling.

Usage:
    python -m pipeline.features data/sample.wav
    python -m pipeline.features clip.wav --transcript "hello how are you"
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

# ---------------------------------------------------------------- contract --

# Order: the five keys already in contract v1.0, then the M3 additions.
FEATURE_KEYS = (
    "speaking_rate",        # words per second of speech time (transcript words / speech seconds)
    "mean_pause_duration",  # mean length in seconds of pauses between speech
    "pitch_mean",           # Hz, mean of voiced f0
    "pitch_variation",      # Hz, standard deviation of voiced f0
    "rms_energy",           # mean frame RMS over speech frames (full-scale = 1.0)
    "pause_ratio",          # fraction of the clip that is not speech (0..1)
    "energy_variation",     # coefficient of variation of frame RMS over speech frames
    "jitter_local",         # Praat local jitter, a fraction (0.01 = 1 %)
    "shimmer_local",        # Praat local shimmer, a fraction (0.05 = 5 %)
    "hnr_db",               # mean harmonics-to-noise ratio in dB
)

CONTRACT_PATH = Path(__file__).resolve().parent.parent / "contract" / "mock_session.json"

# ------------------------------------------------------------------ tuning --

TARGET_SR = 16000
FRAME_S = 0.025            # RMS frame length
HOP_S = 0.010              # RMS hop
MIN_DURATION_S = 0.5       # shorter clips raise ValueError
SILENCE_RMS = 1e-3         # loudest frame below this (about -60 dBFS) = silent clip
MIN_DYNAMIC_RANGE_DB = 6.0 # less spread than this between quiet and loud frames = steady signal
THRESHOLD_FRACTION = 0.3   # speech threshold sits this far up from the noise floor to the speech level
MIN_GAP_S = 0.15           # shorter gaps inside speech are not pauses (stop consonants etc.)
MIN_SPEECH_S = 0.05        # shorter bursts are treated as clicks, not speech
MIN_SPEECH_FOR_RATE_S = 0.3
MIN_SPEECH_FOR_VOICE_S = 0.3
MIN_VOICED_FRAMES = 10
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0
DECIMALS = 5

VOICE_KEYS = ("pitch_mean", "pitch_variation", "jitter_local", "shimmer_local", "hnr_db")


# ----------------------------------------------------------------- helpers --

def _clean(value) -> Optional[float]:
    """float rounded for JSON, or None for None/NaN/inf."""
    if value is None:
        return None
    value = float(value)
    return round(value, DECIMALS) if math.isfinite(value) else None


def _frame_rms(audio: np.ndarray, sr: int) -> np.ndarray:
    frame = int(round(FRAME_S * sr))
    hop = int(round(HOP_S * sr))
    if audio.size < frame:
        audio = np.pad(audio, (0, frame - audio.size))
    csum = np.concatenate(([0.0], np.cumsum(audio ** 2)))
    starts = np.arange(0, audio.size - frame + 1, hop)
    energy = (csum[starts + frame] - csum[starts]) / frame
    return np.sqrt(np.maximum(energy, 0.0))


def _runs(mask: np.ndarray):
    """[(start, end, value), ...] for consecutive equal values; end is exclusive."""
    if mask.size == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [mask.size]))
    return [(int(s), int(e), bool(mask[s])) for s, e in zip(starts, ends)]


def _speech_mask(rms: np.ndarray) -> np.ndarray:
    """True for frames that contain speech.

    The threshold adapts to each clip: it sits between the noise floor (10th
    percentile of frame level) and the speech level (95th percentile). This
    keeps pause detection working on the 10 dB SNR noisy copies, where a fixed
    "30 dB below peak" rule would count the noise as speech.
    """
    if rms.size == 0 or float(rms.max()) < SILENCE_RMS:
        return np.zeros(rms.size, dtype=bool)

    db = 20.0 * np.log10(rms + 1e-10)
    floor = float(np.percentile(db, 10))
    peak = float(np.percentile(db, 95))
    if peak - floor < MIN_DYNAMIC_RANGE_DB:
        return np.ones(rms.size, dtype=bool)   # steady sound, no pauses to find

    mask = db > floor + THRESHOLD_FRACTION * (peak - floor)

    # Close short gaps that sit between speech, then drop very short bursts.
    min_gap = int(round(MIN_GAP_S / HOP_S))
    for start, end, value in _runs(mask):
        if not value and 0 < start and end < mask.size and (end - start) < min_gap:
            mask[start:end] = True
    min_speech = int(round(MIN_SPEECH_S / HOP_S))
    for start, end, value in _runs(mask):
        if value and (end - start) < min_speech:
            mask[start:end] = False
    return mask


def _internal_pauses(mask: np.ndarray) -> list:
    """Lengths in seconds of non-speech runs that have speech on both sides."""
    return [
        (end - start) * HOP_S
        for start, end, value in _runs(mask)
        if not value and start > 0 and end < mask.size
    ]


def _safe(fn):
    """Praat raises on degenerate input (too few periods, etc.): treat as 'not measurable'."""
    try:
        return fn()
    except parselmouth.PraatError:
        return None


def _voice_features(audio: np.ndarray, sr: int) -> dict:
    out = dict.fromkeys(VOICE_KEYS)
    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    def pitch_stats():
        pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if f0.size < MIN_VOICED_FRAMES:
            return None
        return float(np.mean(f0)), float(np.std(f0))

    stats = _safe(pitch_stats)
    if stats is not None:
        out["pitch_mean"], out["pitch_variation"] = stats

    points = _safe(
        lambda: call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
    )
    if points is not None:
        out["jitter_local"] = _safe(
            lambda: call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        )
        out["shimmer_local"] = _safe(
            lambda: call([sound, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        )

    def hnr():
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, PITCH_FLOOR_HZ, 0.1, 1.0)
        return call(harmonicity, "Get mean", 0, 0)

    out["hnr_db"] = _safe(hnr)
    return out


# ---------------------------------------------------------------- contract --

def validate_features(features: dict, expected_keys=FEATURE_KEYS) -> bool:
    """Raise ValueError unless features has exactly the contract keys and valid values."""
    if not isinstance(features, dict):
        raise ValueError("features must be a dict")
    missing = set(expected_keys) - set(features)
    extra = set(features) - set(expected_keys)
    if missing or extra:
        raise ValueError(
            f"features block does not match the contract "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    for key, value in features.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"feature '{key}' must be a finite number or None, got {value!r}")
    return True


def contract_feature_keys(path: Path = CONTRACT_PATH) -> tuple:
    """Feature keys currently in the contract file."""
    with open(path, "r", encoding="utf-8") as f:
        return tuple(json.load(f)["features"].keys())


# ----------------------------------------------------------------- extract --

def extract(wav_path, transcript: Union[str, dict, None] = None) -> dict:
    """Extract the contract `features` block from one audio file.

    transcript: the STT text (or the dict returned by backend.stt.transcribe_audio).
                Needed only for speaking_rate; without it that value is None.

    Raises FileNotFoundError for a missing file and ValueError for audio that
    cannot be decoded or is shorter than MIN_DURATION_S. A silent clip is NOT an
    error: it returns pause_ratio 1.0 and None for features that need speech.
    """
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    except Exception as exc:  # decoder errors vary by backend
        raise ValueError(f"Could not decode audio file: {path.name}") from exc

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError(f"Audio has no valid samples: {path.name}")
    if audio.size / sr < MIN_DURATION_S:
        raise ValueError(
            f"Recording too short ({audio.size / sr:.2f}s); minimum is {MIN_DURATION_S}s"
        )
    audio = audio.astype(np.float64)

    # --- timing and energy (numpy only) ---
    rms = _frame_rms(audio, sr)
    mask = _speech_mask(rms)
    n_frames = int(mask.size)
    n_speech = int(mask.sum())
    speech_time = n_speech * HOP_S

    pause_ratio = 1.0 - n_speech / n_frames

    if n_speech == 0:
        mean_pause = None
    else:
        pauses = _internal_pauses(mask)
        mean_pause = float(np.mean(pauses)) if pauses else 0.0

    if isinstance(transcript, dict):
        transcript = transcript.get("transcript")
    n_words = len(transcript.split()) if isinstance(transcript, str) else 0
    speaking_rate = (
        n_words / speech_time if n_words > 0 and speech_time >= MIN_SPEECH_FOR_RATE_S else None
    )

    speech_rms = rms[mask] if n_speech else rms
    rms_energy = float(np.mean(speech_rms))
    energy_variation = (
        float(np.std(speech_rms) / rms_energy) if n_speech and rms_energy > 0 else None
    )

    features = {key: None for key in FEATURE_KEYS}
    features.update(
        speaking_rate=speaking_rate,
        mean_pause_duration=mean_pause,
        rms_energy=rms_energy,
        pause_ratio=pause_ratio,
        energy_variation=energy_variation,
    )

    # --- pitch and voice quality (Praat) ---
    if speech_time >= MIN_SPEECH_FOR_VOICE_S:
        features.update(_voice_features(audio, sr))

    features = {key: _clean(features[key]) for key in FEATURE_KEYS}
    validate_features(features)
    return features


# --------------------------------------------------------------------- CLI --

def main():
    parser = argparse.ArgumentParser(description="Extract VoxSense features from an audio file.")
    parser.add_argument("audio_path")
    parser.add_argument("--transcript", default=None, help="STT text, used for speaking_rate")
    args = parser.parse_args()
    try:
        print(json.dumps(extract(args.audio_path, args.transcript), indent=2))
    except (FileNotFoundError, ValueError) as error:
        print("Error:", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
