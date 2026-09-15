import argparse
import csv
from pathlib import Path

import yaml


def parse_la_line(line: str) -> dict:
    # LA format: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
    speaker, file_id, _, system_id, key = line.strip().split()
    return {
        "speaker_id": speaker,
        "file_id": file_id,
        "system_id": system_id,
        "attack_id": "",
        "env_id": "",
        "label": 0 if key == "bonafide" else 1,
        "key": key,
    }


def parse_pa_line(line: str) -> dict:
    # PA format: SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT_ID ATTACK_ID KEY
    speaker, file_id, env_id, attack_id, key = line.strip().split()
    return {
        "speaker_id": speaker,
        "file_id": file_id,
        "system_id": "",
        "attack_id": attack_id,
        "env_id": env_id,
        "label": 0 if key == "bonafide" else 1,
        "key": key,
    }


def build_manifest(data_root: Path, out_csv: Path, scenario: str, split: str, verify_exists: bool) -> dict:
    if scenario == "LA":
        protocol = data_root / f"LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.{'trn' if split == 'train' else 'trl'}.txt"
        audio_dir = data_root / f"LA/LA/ASVspoof2019_LA_{split}/flac"
        parser = parse_la_line
    else:
        protocol = data_root / f"PA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.{split}.{'trn' if split == 'train' else 'trl'}.txt"
        audio_dir = data_root / f"PA/PA/ASVspoof2019_PA_{split}/flac"
        parser = parse_pa_line

    if not protocol.exists():
        raise FileNotFoundError(f"Missing protocol: {protocol}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio directory: {audio_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    missing = 0

    with protocol.open("r", encoding="utf-8") as r, out_csv.open("w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(
            w,
            fieldnames=[
                "relative_path",
                "label",
                "speaker_id",
                "scenario",
                "split",
                "system_id",
                "attack_id",
                "env_id",
                "snr_type",
                "snr_level",
                "key",
            ],
        )
        writer.writeheader()

        for line in r:
            line = line.strip()
            if not line:
                continue
            row = parser(line)
            rel = Path(audio_dir.relative_to(data_root)) / f"{row['file_id']}.flac"
            abs_path = data_root / rel
            if verify_exists and not abs_path.exists():
                missing += 1

            writer.writerow(
                {
                    "relative_path": str(rel),
                    "label": row["label"],
                    "speaker_id": row["speaker_id"],
                    "scenario": scenario,
                    "split": split,
                    "system_id": row["system_id"],
                    "attack_id": row["attack_id"],
                    "env_id": row["env_id"],
                    "snr_type": "none",
                    "snr_level": "clean",
                    "key": row["key"],
                }
            )
            total += 1

    return {
        "manifest": str(out_csv),
        "protocol": str(protocol),
        "scenario": scenario,
        "split": split,
        "rows": total,
        "missing_audio": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 1 manifests from ASVspoof2019 CM protocols.")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="ASVspoof root path")
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"), help="Output directory")
    parser.add_argument("--verify-exists", action="store_true", help="Check each protocol entry has audio file")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"manifests": []}
    for scenario in ("LA", "PA"):
        for split in ("train", "dev", "eval"):
            out_csv = args.out_dir / f"{scenario.lower()}_{split}.csv"
            info = build_manifest(args.data_root, out_csv, scenario, split, args.verify_exists)
            summary["manifests"].append(info)

    # Per problem statement: Set A = LA, Set B = PA.
    runs = {
        "run1": {
            "description": "Train/validate on Set A (LA), evaluate on LA (in-domain) and PA (cross-domain)",
            "train_manifest": str(args.out_dir / "la_train.csv"),
            "val_manifest": str(args.out_dir / "la_dev.csv"),
            "test_in_manifest": str(args.out_dir / "la_eval.csv"),
            "test_cross_manifest": str(args.out_dir / "pa_eval.csv"),
        },
        "run2": {
            "description": "Train/validate on Set B (PA), evaluate on PA (in-domain) and LA (cross-domain)",
            "train_manifest": str(args.out_dir / "pa_train.csv"),
            "val_manifest": str(args.out_dir / "pa_dev.csv"),
            "test_in_manifest": str(args.out_dir / "pa_eval.csv"),
            "test_cross_manifest": str(args.out_dir / "la_eval.csv"),
        },
    }

    with (args.out_dir / "runs.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(runs, f, sort_keys=False)

    with (args.out_dir / "summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"Generated manifests in: {args.out_dir}")
    for item in summary["manifests"]:
        print(item)


if __name__ == "__main__":
    main()
