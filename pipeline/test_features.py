"""M3 tests. Fully offline: all audio is synthesised here, no API or dataset needed.

Run from the repo root:   pytest pipeline/ -v
"""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from pipeline.features import (
    CONTRACT_PATH,
    FEATURE_KEYS,
    contract_feature_keys,
    extract,
    validate_features,
)

SR = 16000


# ----------------------------------------------------------- synthetic audio --

def voiced(duration, sr=SR, f0=120.0):
    """Clean voiced-like sound: 8 harmonics of f0 with 20 ms fade in/out."""
    t = np.arange(int(duration * sr)) / sr
    sig = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 9))
    sig = 0.3 * sig / np.max(np.abs(sig))
    ramp = int(0.02 * sr)
    env = np.ones_like(sig)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return sig * env


def silence(duration, sr=SR):
    return np.zeros(int(duration * sr))


def speech_like(sr=SR):
    """6 s clip: 1 s voiced, 1 s silence, repeated 3 times.
    Speech = 3 s, internal pauses = two of 1 s, trailing silence = 1 s."""
    return np.concatenate([voiced(1, sr), silence(1, sr)] * 3)


def add_noise(audio, snr_db=10, seed=42):
    rng = np.random.default_rng(seed)
    signal_rms = np.sqrt(np.mean(audio ** 2))
    noise = rng.standard_normal(audio.shape)
    noise *= signal_rms / (10 ** (snr_db / 20)) / np.sqrt(np.mean(noise ** 2))
    return audio + noise


def write_wav(path, audio, sr=SR):
    wavfile.write(str(path), sr, audio.astype(np.float32))
    return path


# ------------------------------------------------------------------ contract --

def test_output_has_exact_contract_keys_and_is_json(tmp_path):
    f = extract(write_wav(tmp_path / "a.wav", speech_like()), transcript="one two three")
    assert tuple(f.keys()) == FEATURE_KEYS
    json.dumps(f, allow_nan=False)          # no NaN / inf anywhere


def test_feature_keys_match_contract_file():
    assert CONTRACT_PATH.is_file()
    assert set(FEATURE_KEYS) == set(contract_feature_keys())


def test_validate_rejects_missing_extra_and_nan_values():
    good = {k: 1.0 for k in FEATURE_KEYS}
    assert validate_features(good)
    with pytest.raises(ValueError):
        validate_features({k: v for k, v in good.items() if k != "hnr_db"})
    with pytest.raises(ValueError):
        validate_features({**good, "surprise": 1.0})
    with pytest.raises(ValueError):
        validate_features({**good, "pause_ratio": float("nan")})


# --------------------------------------------------------- timing and energy --

def test_silent_clip_has_high_pause_ratio(tmp_path):
    f = extract(write_wav(tmp_path / "silent.wav", silence(2)), transcript="hello there")
    assert f["pause_ratio"] >= 0.95
    assert f["speaking_rate"] is None
    assert f["mean_pause_duration"] is None
    assert f["pitch_mean"] is None
    assert f["rms_energy"] == pytest.approx(0.0, abs=1e-6)


def test_speech_like_clip_pause_ratio_and_mean_pause(tmp_path):
    f = extract(write_wav(tmp_path / "s.wav", speech_like()))
    assert 0.40 <= f["pause_ratio"] <= 0.60           # true value is 0.5
    assert 0.85 <= f["mean_pause_duration"] <= 1.15   # two internal pauses of 1 s
    assert f["rms_energy"] > 0.05


def test_continuous_speech_has_low_pause_ratio(tmp_path):
    f = extract(write_wav(tmp_path / "c.wav", voiced(3)))
    assert f["pause_ratio"] <= 0.1
    assert f["mean_pause_duration"] == 0.0


def test_speaking_rate_is_words_over_speech_time(tmp_path):
    # 6 words over about 3 s of speech is about 2 words per second
    f = extract(write_wav(tmp_path / "r.wav", speech_like()), transcript="one two three four five six")
    assert 1.7 <= f["speaking_rate"] <= 2.3


def test_speaking_rate_ignores_pauses(tmp_path):
    # same speech, 4x the silence: rate must not change much
    base = extract(write_wav(tmp_path / "x.wav", speech_like()), transcript="a b c d e f")
    padded = np.concatenate([speech_like(), silence(12)])
    slow = extract(write_wav(tmp_path / "y.wav", padded), transcript="a b c d e f")
    assert slow["speaking_rate"] == pytest.approx(base["speaking_rate"], rel=0.1)
    assert slow["pause_ratio"] > base["pause_ratio"]


def test_no_transcript_means_no_speaking_rate(tmp_path):
    f = extract(write_wav(tmp_path / "n.wav", speech_like()))
    assert f["speaking_rate"] is None
    assert extract(write_wav(tmp_path / "m.wav", speech_like()), transcript="")["speaking_rate"] is None


def test_stt_result_dict_is_accepted_as_transcript(tmp_path):
    p = write_wav(tmp_path / "d.wav", speech_like())
    a = extract(p, transcript="one two three four five six")
    b = extract(p, transcript={"transcript": "one two three four five six", "detected_language": "hi-IN"})
    assert a["speaking_rate"] == b["speaking_rate"]


def test_energy_variation_higher_when_loudness_changes(tmp_path):
    steady = extract(write_wav(tmp_path / "st.wav", voiced(3)))
    varying = np.concatenate([voiced(1) * 0.25, voiced(1), voiced(1) * 0.5])
    var = extract(write_wav(tmp_path / "va.wav", varying))
    assert var["energy_variation"] > steady["energy_variation"]


# ------------------------------------------------------ pitch / voice quality --

def test_pitch_and_voice_quality_on_clean_voiced_clip(tmp_path):
    f = extract(write_wav(tmp_path / "p.wav", speech_like()))
    assert f["pitch_mean"] == pytest.approx(120.0, abs=6.0)
    assert f["pitch_variation"] < 15.0
    assert f["jitter_local"] is not None and f["jitter_local"] < 0.02
    assert f["shimmer_local"] is not None and f["shimmer_local"] < 0.10
    assert f["hnr_db"] is not None and f["hnr_db"] > 15.0


# --------------------------------------------------- degraded clips, errors --

def test_noisy_10db_clip_does_not_crash_and_still_finds_pauses(tmp_path):
    f = extract(write_wav(tmp_path / "noisy.wav", add_noise(speech_like(), 10)))
    assert tuple(f.keys()) == FEATURE_KEYS
    assert 0.35 <= f["pause_ratio"] <= 0.65
    json.dumps(f, allow_nan=False)


def test_8khz_clip_works(tmp_path):
    f = extract(write_wav(tmp_path / "lo.wav", speech_like(sr=8000), sr=8000))
    assert tuple(f.keys()) == FEATURE_KEYS
    assert 0.35 <= f["pause_ratio"] <= 0.65


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract(tmp_path / "nope.wav")


def test_too_short_clip_raises(tmp_path):
    with pytest.raises(ValueError):
        extract(write_wav(tmp_path / "short.wav", voiced(0.2)))


def test_corrupt_file_raises_value_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(ValueError):
        extract(bad)
