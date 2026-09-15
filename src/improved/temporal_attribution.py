from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.improved.explainability_utils import (
    clean_output_dir,
    ensure_dir,
    get_score_column,
    get_top_n_files,
    infer_aasist,
    load_aasist,
    load_audio,
    log_error,
    write_manifest,
    write_summary,
)


def _select_utt_ids(scores_csv: str | Path, top_n: int, utt_id: str | None) -> list[str]:
    if utt_id:
        return [utt_id]
    if top_n and top_n > 0:
        return get_top_n_files(scores_csv, top_n)
    df = pd.read_csv(scores_csv)
    return df["utt_id"].astype(str).tolist()


def _load_scores(scores_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    if "utt_id" not in df.columns:
        raise ValueError("scores_csv must contain utt_id")
    return df


def _resolve_audio_path(input_dir: Path, utt_id: str) -> Path | None:
    patterns = [f"**/*{utt_id}*.wav", f"**/*{utt_id}*.flac", f"**/{utt_id}.wav", f"**/{utt_id}.flac"]
    for pattern in patterns:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _mask_window(audio: torch.Tensor, start: int, end: int, mask_type: str) -> torch.Tensor:
    masked = audio.clone()
    if mask_type == "silence":
        masked[start:end] = 0.0
        return masked
    noise = torch.randn(end - start, dtype=audio.dtype) * (audio[start:end].std().clamp_min(1e-6))
    masked[start:end] = noise
    return masked


def _plot_file(output_path: Path, waveform: np.ndarray, sr: int, times: np.ndarray, deltas: np.ndarray, score: float, label: str, utt_id: str, top_segments: list[tuple[float, float, float]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    try:
        t = np.arange(waveform.shape[0]) / sr
        axes[0].plot(t, waveform, linewidth=0.8)
        axes[0].set_title(f"{utt_id} | AASIST score={score:.4f} | {label}")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")
        for seg_start, seg_end, delta in top_segments:
            axes[0].axvspan(seg_start, seg_end, color="red", alpha=0.12)
            axes[0].text(seg_start, 0.9 * waveform.max() if waveform.max() != 0 else 0.1, f"{seg_start:.2f}s", rotation=90, fontsize=8)

        colors = ["red" if d >= 0 else "blue" for d in deltas]
        axes[1].bar(times, deltas, width=np.diff(np.r_[times, times[-1] + (times[1] - times[0] if len(times) > 1 else 0.025)]), color=colors, alpha=0.85)
        axes[1].axhline(0.0, linestyle="--", color="black", linewidth=1)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Delta score (original - masked)")
        axes[1].set_title("Temporal attribution (red=fake pushing, blue=bonafide pushing)")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal segment attribution explainability")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--aasist-config", type=Path, required=True)
    parser.add_argument("--aasist-checkpoint", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("explainability/temporal_attribution"))
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--stride-ms", type=float, default=25.0)
    parser.add_argument("--mask-type", choices=["silence", "noise"], default="silence")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--utt-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.time()
    ensure_dir(args.output_dir)
    clean_output_dir(args.output_dir)
    model = load_aasist(args.aasist_config, args.aasist_checkpoint, device=args.device)
    score_df = _load_scores(args.scores_csv)
    score_col = get_score_column(args.scores_csv)
    utt_ids = _select_utt_ids(args.scores_csv, args.top_n, args.utt_id)

    segment_rows = []
    output_paths: list[str] = []
    files_processed = 0
    files_failed = 0
    files_skipped = 0
    score_accum: list[float] = []
    aggregate_bins = np.linspace(0.0, 1.0, 101)
    aggregate_vals: list[np.ndarray] = []

    for utt in tqdm(utt_ids, desc="temporal_attribution"):
        audio_path = _resolve_audio_path(args.input_dir, utt)
        if audio_path is None:
            files_skipped += 1
            log_error("temporal_attribution", f"missing audio for {utt}")
            continue
        try:
            waveform, sr = load_audio(audio_path)
            row = score_df[score_df["utt_id"].astype(str) == str(utt)]
            if row.empty:
                raise ValueError(f"utt_id missing from scores csv: {utt}")
            score = float(row.iloc[0][score_col])
            label = "FAKE" if ("label" in row.columns and int(row.iloc[0]["label"]) == 1) or score >= 0.5 else "BONAFIDE"

            win = int((args.window_ms / 1000.0) * sr)
            stride = max(1, int((args.stride_ms / 1000.0) * sr))
            if win <= 0:
                raise ValueError("window_ms too small")
            starts = list(range(0, max(1, waveform.shape[0] - win + 1), stride))
            if starts and starts[-1] + win < waveform.shape[0]:
                starts.append(max(0, waveform.shape[0] - win))
            if not starts:
                starts = [0]

            deltas = []
            centers = []
            local_rows = []
            for start_idx in starts:
                end_idx = min(start_idx + win, waveform.shape[0])
                masked = _mask_window(waveform, start_idx, end_idx, args.mask_type)
                perturbed = infer_aasist(model, masked, sr)
                delta = score - perturbed
                deltas.append(delta)
                center_sec = ((start_idx + end_idx) / 2.0) / sr
                centers.append(center_sec)
                local_rows.append((start_idx, end_idx, delta))
                segment_rows.append(
                    {
                        "utt_id": utt,
                        "segment_start_ms": float(start_idx / sr * 1000.0),
                        "segment_end_ms": float(end_idx / sr * 1000.0),
                        "delta_score": float(delta),
                        "mask_type": args.mask_type,
                    }
                )

            top_segments = sorted(local_rows, key=lambda x: x[2], reverse=True)[:3]
            out_path = args.output_dir / f"{utt}_temporal.png"
            _plot_file(out_path, waveform.numpy(), sr, np.array(centers), np.array(deltas), score, label, utt, [((s / sr), (e / sr), d) for s, e, d in top_segments])
            output_paths.append(str(out_path))
            files_processed += 1
            score_accum.append(score)

            rel_pos = np.array([c / max(1e-9, waveform.shape[0] / sr) for c in centers])
            interp = np.interp(aggregate_bins, rel_pos, np.array(deltas), left=np.nan, right=np.nan)
            aggregate_vals.append(interp)
        except Exception as exc:
            files_failed += 1
            log_error("temporal_attribution", f"{utt}: {exc}")

    if aggregate_vals:
        arr = np.vstack(aggregate_vals)
        mean_profile = np.nanmean(arr, axis=0)
        fig = plt.figure(figsize=(10, 4))
        try:
            plt.plot(aggregate_bins, mean_profile, linewidth=2)
            plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
            plt.xlabel("Relative time (0-1)")
            plt.ylabel("Mean delta score")
            plt.title("Aggregate temporal attribution profile")
            fig.savefig(args.output_dir / "aggregate_temporal_profile.png", dpi=180)
        finally:
            plt.close(fig)

    seg_csv = args.output_dir / "segment_scores.csv"
    pd.DataFrame(segment_rows).to_csv(seg_csv, index=False)
    output_paths.extend([str(seg_csv), str(args.output_dir / "aggregate_temporal_profile.png") if aggregate_vals else ""])
    write_manifest(args.output_dir, [p for p in output_paths if p])
    write_summary(
        "temporal_attribution",
        args.output_dir,
        files_processed,
        files_failed,
        files_skipped,
        float(np.mean(score_accum)) if score_accum else 0.0,
        time.time() - start,
    )


if __name__ == "__main__":
    main()
