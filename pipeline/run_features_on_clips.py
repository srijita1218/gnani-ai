"""Run extract() on every M1 clip (clean, 8 kHz and noisy) and report crashes.

    python -m pipeline.run_features_on_clips
    python -m pipeline.run_features_on_clips data/raw/recordings data/degraded

Default folders: data/raw/recordings and data/degraded (searched recursively),
plus data/sample.wav if present. The per-clip table is printed to the terminal
only. The saved summary (eval/results/features_check.json) holds counts and
failures, no per-clip feature values, so personal recordings stay private.

Exit code 0 = no crashes and at least 10 clips succeeded.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline.features import FEATURE_KEYS, extract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "recordings",
    PROJECT_ROOT / "data" / "degraded",
    PROJECT_ROOT / "data" / "sample.wav",
]
EXTENSIONS = {".wav", ".mp3", ".flac"}
SUMMARY_PATH = PROJECT_ROOT / "eval" / "results" / "features_check.json"
MIN_CLIPS = 10


def find_clips(paths):
    clips = set()
    for p in map(Path, paths):
        if p.is_file() and p.suffix.lower() in EXTENSIONS:
            clips.add(p)
        elif p.is_dir():
            clips.update(f for f in p.rglob("*") if f.suffix.lower() in EXTENSIONS)
    return sorted(clips)


def condition(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    if "noisy" in parts:
        return "noisy_10db"
    if "8khz" in parts:
        return "8khz"
    return "clean"


def fmt(value, width=8):
    return f"{value:>{width}.3f}" if isinstance(value, (int, float)) else f"{'-':>{width}}"


def main():
    parser = argparse.ArgumentParser(description="Run feature extraction on all clips.")
    parser.add_argument("paths", nargs="*", help="files or folders (default: M1 data folders)")
    args = parser.parse_args()

    clips = find_clips(args.paths or DEFAULT_PATHS)
    if not clips:
        print("No audio clips found. Check the folders exist (data/ is not in git).")
        raise SystemExit(1)

    ok = Counter()
    failed = []
    none_counts = Counter()

    print(f"{'clip':<42}{'cond':<12}{'pause':>8}{'rate':>8}{'pitch':>8}{'jitter':>8}{'hnr':>8}")
    for clip in clips:
        cond = condition(clip)
        try:
            f = extract(clip)
        except Exception as exc:  # report every crash, don't stop
            failed.append({"file": clip.name, "condition": cond, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{clip.name[:40]:<42}{cond:<12}FAILED  {type(exc).__name__}: {exc}")
            continue
        ok[cond] += 1
        for key in FEATURE_KEYS:
            if f[key] is None:
                none_counts[key] += 1
        print(
            f"{clip.name[:40]:<42}{cond:<12}{fmt(f['pause_ratio'])}{fmt(f['speaking_rate'])}"
            f"{fmt(f['pitch_mean'])}{fmt(f['jitter_local'], 8)}{fmt(f['hnr_db'])}"
        )

    succeeded = sum(ok.values())
    summary = {
        "total_clips": len(clips),
        "succeeded": succeeded,
        "failed": len(failed),
        "succeeded_by_condition": dict(ok),
        "none_counts_per_feature": dict(none_counts),
        "failures": failed,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{succeeded}/{len(clips)} succeeded, {len(failed)} failed.")
    print("By condition:", dict(ok))
    print("Features that were None (count of clips):", dict(none_counts) or "none")
    print("Summary saved to", SUMMARY_PATH)

    if failed or succeeded < MIN_CLIPS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
