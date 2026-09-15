from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
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


def _band_filter(audio: np.ndarray, sr: int, low: float, high: float, order: int = 6) -> np.ndarray:
    nyq = sr / 2.0
    if low <= 0 and high >= nyq:
        return audio
    if low <= 0:
        sos = butter(order, high / nyq, btype="lowpass", output="sos")
    elif high >= nyq:
        sos = butter(order, low / nyq, btype="highpass", output="sos")
    else:
        sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _score_band(model, audio: np.ndarray, sr: int, band: tuple[float, float], target_rms: float) -> float:
    low, high = band
    filtered = _band_filter(audio, sr, low, high)
    filtered_rms = _rms(filtered)
    if filtered_rms > 0:
        filtered = filtered * (target_rms / filtered_rms)
    return infer_aasist(model, torch_from_numpy(filtered), sr)


def torch_from_numpy(audio: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def _plot_bars(output_path: Path, utt_id: str, band_labels: list[str], scores: list[float], label: str) -> None:
    fig = plt.figure(figsize=(12, 6))
    try:
        colors = ["red" if s > 0.5 else "green" for s in scores]
        bars = plt.bar(band_labels, scores, color=colors, alpha=0.85)
        plt.axhline(0.5, linestyle="--", color="black", linewidth=1)
        plt.ylim(0.0, 1.0)
        plt.ylabel("AASIST spoof probability")
        plt.title(f"{utt_id} | {label}")
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width() / 2.0, score + 0.02, f"{score:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)


def _analysis_for_dir(model, input_dir: Path, utt_ids: list[str], band_edges: list[float], output_dir: Path, score_df: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, Counter, list[float], list[str], int, int, int]:
    ensure_dir(output_dir)
    rows = []
    band_counter: Counter[str] = Counter()
    files_processed = files_failed = files_skipped = 0
    score_accum: list[float] = []
    band_labels = [f"{int(band_edges[i])}-{int(band_edges[i+1])}Hz" for i in range(len(band_edges) - 1)] + ["full"]
    score_cols = [f"{label}_score" for label in band_labels]
    score_cols = [f"{label}_score" for label in band_labels]

    for utt in tqdm(utt_ids, desc=f"subband:{input_dir.name}"):
        audio_path = _resolve_audio_path(input_dir, utt)
        if audio_path is None:
            files_skipped += 1
            log_error("subband_decomposition", f"missing audio for {utt} in {input_dir}")
            continue
        try:
            waveform, sr = load_audio(audio_path)
            audio = waveform.numpy()
            target_rms = _rms(audio)
            meta = score_df[score_df["utt_id"].astype(str) == str(utt)]
            if meta.empty:
                raise ValueError(f"utt_id missing from scores csv: {utt}")
            full_score = float(meta.iloc[0][score_col])
            scores = []
            for i, (low, high) in enumerate(zip(band_edges[:-1], band_edges[1:])):
                if i == 0:
                    band_score = _score_band(model, audio, sr, (0, high), target_rms)
                elif i == len(band_edges) - 2:
                    band_score = _score_band(model, audio, sr, (low, sr / 2.0), target_rms)
                else:
                    band_score = _score_band(model, audio, sr, (low, high), target_rms)
                scores.append(band_score)
            scores.append(full_score)
            best_band = band_labels[int(np.argmax(scores))]
            band_counter[best_band] += 1
            row = {"utt_id": utt, "label": int(full_score >= 0.5)}
            for col_name, value in zip(score_cols, scores):
                row[col_name] = float(value)
            rows.append(row)
            out_path = output_dir / f"{utt}_subband.png"
            title_label = "FAKE" if full_score > 0.5 else "BONAFIDE"
            _plot_bars(out_path, utt, band_labels, scores, title_label)
            files_processed += 1
            score_accum.append(full_score)
        except Exception as exc:
            files_failed += 1
            log_error("subband_decomposition", f"{utt}: {exc}")

    aggregate_df = pd.DataFrame(rows)
    return aggregate_df, band_counter, score_accum, band_labels, files_processed, files_failed, files_skipped


def _comparison_plot(output_path: Path, mean_by_dir: dict[str, list[float]], band_labels: list[str]) -> None:
    fig = plt.figure(figsize=(14, 7))
    try:
        x = np.arange(len(band_labels))
        for dir_name, values in mean_by_dir.items():
            plt.plot(x, values, marker="o", linewidth=2, label=dir_name)
        plt.xticks(x, band_labels, rotation=30, ha="right")
        plt.ylabel("Mean AASIST spoof probability")
        plt.title("Sub-band score comparison across pipeline stages")
        plt.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-band score decomposition explainability")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--aasist-config", type=Path, required=True)
    parser.add_argument("--aasist-checkpoint", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("explainability/subband_decomposition"))
    parser.add_argument("--bands", type=str, default="0,500,1000,2000,4000,6000,8000")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--utt-id", type=str, default=None)
    parser.add_argument("--compare-dirs", type=Path, nargs="*", default=None)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.time()
    ensure_dir(args.output_dir)
    clean_output_dir(args.output_dir)
    model = load_aasist(args.aasist_config, args.aasist_checkpoint, device=args.device)
    score_df = _load_scores(args.scores_csv)
    score_col = get_score_column(args.scores_csv)
    utt_ids = _select_utt_ids(args.scores_csv, args.top_n, args.utt_id)
    band_edges = [float(x.strip()) for x in args.bands.split(",") if x.strip()]
    if len(band_edges) < 3:
        raise ValueError("bands must contain at least 3 edges")
    band_labels = [f"{int(band_edges[i])}-{int(band_edges[i+1])}Hz" for i in range(len(band_edges) - 1)] + ["full"]
    score_cols = [f"{label}_score" for label in band_labels]

    input_runs = [args.input_dir] + (args.compare_dirs or [])
    comparison_mean = {}
    output_paths: list[str] = []
    overall_rows = []
    files_processed = files_failed = files_skipped = 0
    score_accum: list[float] = []

    for idx, run_dir in enumerate(input_runs):
        run_name = run_dir.name if idx == 0 else f"compare_{run_dir.name}"
        run_out = args.output_dir / run_name
        ensure_dir(run_out)
        aggregate_df, band_counter, score_vals, _, proc, fail, skip = _analysis_for_dir(
            model,
            run_dir,
            utt_ids,
            band_edges,
            run_out,
            score_df,
            score_col,
        )
        files_processed += proc
        files_failed += fail
        files_skipped += skip
        score_accum.extend(score_vals)
        if not aggregate_df.empty:
            aggregate_df.to_csv(run_out / "aggregate_subband_scores.csv", index=False)
            output_paths.append(str(run_out / "aggregate_subband_scores.csv"))
            melted = aggregate_df.melt(id_vars=["utt_id", "label"], var_name="band", value_name="score")
            fig = plt.figure(figsize=(14, 7))
            try:
                melted.boxplot(column="score", by="band", grid=False, rot=30)
                plt.suptitle("")
                plt.title(f"Sub-band score distribution: {run_name}")
                plt.ylabel("AASIST spoof probability")
                fig.tight_layout()
                fig.savefig(run_out / "aggregate_subband_boxplot.png", dpi=180)
            finally:
                plt.close(fig)
            output_paths.append(str(run_out / "aggregate_subband_boxplot.png"))

            most = aggregate_df[score_cols].idxmax(axis=1)
            local_counts = Counter([x.replace("_score", "") for x in most.tolist()])
            (run_out / "most_discriminative_band.txt").write_text(
                "Most discriminative band counts:\n"
                + "\n".join(f"{band}: {count}" for band, count in local_counts.items())
                + "\n",
                encoding="utf-8",
            )
            output_paths.append(str(run_out / "most_discriminative_band.txt"))
            for _, row in aggregate_df.iterrows():
                overall_rows.append(row.to_dict() | {"run_name": run_name})

            comparison_mean[run_name] = [float(aggregate_df[col].mean()) for col in score_cols]

    if len(comparison_mean) > 1:
        comparison_csv = args.output_dir / "comparison_subband_scores.csv"
        pd.DataFrame(
            [{"run_name": k, **{band: v for band, v in zip(band_labels, vals)}} for k, vals in comparison_mean.items()]
        ).to_csv(comparison_csv, index=False)
        _comparison_plot(args.output_dir / "comparison_subband_scores.png", comparison_mean, band_labels)
        output_paths.extend([str(comparison_csv), str(args.output_dir / "comparison_subband_scores.png")])

    if overall_rows:
        pd.DataFrame(overall_rows).to_csv(args.output_dir / "all_runs_aggregate_subband_scores.csv", index=False)
        output_paths.append(str(args.output_dir / "all_runs_aggregate_subband_scores.csv"))

    write_manifest(args.output_dir, output_paths)
    write_summary(
        "subband_decomposition",
        args.output_dir,
        files_processed,
        files_failed,
        files_skipped,
        float(np.mean(score_accum)) if score_accum else 0.0,
        time.time() - start,
    )


if __name__ == "__main__":
    main()
