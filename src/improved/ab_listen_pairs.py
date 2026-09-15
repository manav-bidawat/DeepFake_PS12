from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def _run_player(player: str, path: Path, volume: int) -> None:
    if player == "ffplay":
        cmd = [
            "ffplay",
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-volume",
            str(volume),
            str(path),
        ]
    elif player == "aplay":
        cmd = ["aplay", str(path)]
    else:
        raise ValueError(f"Unsupported player: {player}")

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Player failed ({player}) for file: {path}")


def _resolve_csv_path(csv_path: Path, root: Path) -> Path:
    # 1) As provided (absolute or relative to current working directory)
    if csv_path.exists():
        return csv_path.resolve()

    # 2) Relative to --root
    candidate = (root / csv_path).resolve()
    if candidate.exists():
        return candidate

    # 3) Common default location inside processed runs
    if csv_path.name == "listen_samples.csv":
        processed_dir = (root / "processed").resolve()
        if processed_dir.exists():
            matches = sorted(processed_dir.glob("*/listen_samples.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                return matches[0]

    raise FileNotFoundError(f"CSV not found: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play before/after audio pairs for A/B listening.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("processed/redteam_pa_subset_run2/listen_samples.csv"),
        help="CSV containing relative_path_before and relative_path_after columns",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Root path used to resolve relative audio paths")
    parser.add_argument("--player", choices=["ffplay", "aplay"], default="ffplay")
    parser.add_argument("--limit", type=int, default=5, help="How many pairs to play")
    parser.add_argument("--pause-sec", type=float, default=1.0, help="Pause between before and after clips")
    parser.add_argument("--volume", type=int, default=100, help="ffplay volume 0-100")
    parser.add_argument("--list-only", action="store_true", help="Print planned playback order only")
    args = parser.parse_args()

    args.csv = _resolve_csv_path(args.csv, args.root)

    rows = []
    with args.csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No rows found in CSV: {args.csv}")

    count = min(args.limit, len(rows)) if args.limit > 0 else len(rows)
    selected = rows[:count]

    for idx, row in enumerate(selected, start=1):
        utt = row.get("utt_id", f"pair_{idx}")
        before_rel = row.get("relative_path_before", "")
        after_rel = row.get("relative_path_after", "")
        before = (args.root / before_rel).resolve()
        after = (args.root / after_rel).resolve()

        if not before.exists() or not after.exists():
            print(f"[{idx}/{count}] {utt} SKIP missing file(s)")
            continue

        print(f"\n[{idx}/{count}] {utt}")
        print(f"  before: {before}")
        print(f"  after : {after}")

        if args.list_only:
            continue

        print("  playing BEFORE...")
        _run_player(args.player, before, args.volume)
        time.sleep(args.pause_sec)
        print("  playing AFTER...")
        _run_player(args.player, after, args.volume)
        time.sleep(args.pause_sec)

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)