from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _score_column(scores_csv: str | Path) -> str:
    return get_score_column(scores_csv)


def _load_scores(scores_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    if "utt_id" not in df.columns:
        raise ValueError("scores_csv must contain utt_id")
    return df


def _select_utt_ids(scores_csv: str | Path, top_n: int, utt_id: str | None) -> list[str]:
    if utt_id:
        return [utt_id]
    if top_n and top_n > 0:
        return get_top_n_files(scores_csv, top_n)
    return _load_scores(scores_csv)["utt_id"].astype(str).tolist()


def _resolve_audio_path(input_dir: Path, utt_id: str) -> Path | None:
    patterns = [f"**/*{utt_id}*.wav", f"**/*{utt_id}*.flac", f"**/{utt_id}.wav", f"**/{utt_id}.flac"]
    for pattern in patterns:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _original_metadata(df: pd.DataFrame, utt_id: str, score_col: str) -> tuple[float, str]:
    row = df[df["utt_id"].astype(str) == str(utt_id)]
    if row.empty:
        return 0.0, "UNKNOWN"
    score = float(row.iloc[0][score_col])
    if "label" in row.columns:
        label = "FAKE" if int(row.iloc[0]["label"]) == 1 else "BONAFIDE"
    elif score >= 0.5:
        label = "FAKE"
    else:
        label = "BONAFIDE"
    return score, label


def _mel_and_audio(waveform: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int):
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    return mel


def _occlusion_attribution(model, waveform: np.ndarray, sr: int, score: float, block_mel: int, block_time: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_fft = 512
    hop_length = 160
    n_mels = 80
    mel = _mel_and_audio(waveform, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    n_mel_frames = mel.shape[1]
    blocks_mel = math.ceil(n_mels / block_mel)
    blocks_time = math.ceil(n_mel_frames / block_time)
    attribution = np.zeros((blocks_mel, blocks_time), dtype=np.float32)

    for fm in range(blocks_mel):
        f0 = fm * block_mel
        f1 = min((fm + 1) * block_mel, n_mels)
        for tm in range(blocks_time):
            t0 = tm * block_time
            t1 = min((tm + 1) * block_time, n_mel_frames)
            occluded = mel.copy()
            occluded[f0:f1, t0:t1] = 0.0
            reconstructed = librosa.feature.inverse.mel_to_audio(
                occluded,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_iter=8,
                power=2.0,
            )
            perturbed = infer_aasist(model, torch_from_numpy(reconstructed), sr)
            attribution[fm, tm] = score - perturbed
    return mel, attribution, mel_to_db(mel)


def torch_from_numpy(audio: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def mel_to_db(mel: np.ndarray) -> np.ndarray:
    return librosa.power_to_db(np.maximum(mel, 1e-10), ref=np.max)


def _save_figure(
    output_path: Path,
    mel_db: np.ndarray,
    attribution_blocks: np.ndarray,
    sr: int,
    hop_length: int,
    score: float,
    label: str,
    utt_id: str,
    block_mel: int,
    block_time: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    try:
        duration = mel_db.shape[1] * hop_length / sr
        extent = [0, duration, 0, sr / 2]
        axes[0].imshow(mel_db, origin="lower", aspect="auto", cmap="viridis", extent=extent)
        axes[0].set_title(f"{utt_id} | AASIST score={score:.4f} | {label} | Raw Mel Spectrogram")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Frequency (Hz)")

        expanded = np.repeat(np.repeat(attribution_blocks, block_mel, axis=0), block_time, axis=1)
        expanded = expanded[: mel_db.shape[0], : mel_db.shape[1]]
        max_abs = float(np.max(np.abs(expanded)) + 1e-8)
        axes[1].imshow(mel_db, origin="lower", aspect="auto", cmap="viridis", extent=extent)
        im = axes[1].imshow(
            expanded,
            origin="lower",
            aspect="auto",
            cmap="bwr",
            alpha=0.5,
            vmin=-max_abs,
            vmax=max_abs,
            extent=extent,
        )
        axes[1].set_title("Attribution heatmap overlay: red=FAKE pushing, blue=BONAFIDE pushing")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=axes[1], label="Attribution (original - perturbed)")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spectrogram overlay heatmap explainability")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--aasist-config", type=Path, required=True)
    parser.add_argument("--aasist-checkpoint", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("explainability/spectrogram_heatmap"))
    parser.add_argument("--utt-id", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--block-mel", type=int, default=8)
    parser.add_argument("--block-time", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    start = time.time()
    ensure_dir(args.output_dir)
    clean_output_dir(args.output_dir)
    model = load_aasist(args.aasist_config, args.aasist_checkpoint, device=args.device)
    score_df = _load_scores(args.scores_csv)
    score_col = _score_column(args.scores_csv)
    utt_ids = _select_utt_ids(args.scores_csv, args.top_n, args.utt_id)

    output_paths: list[str] = []
    files_processed = 0
    files_failed = 0
    files_skipped = 0
    score_accum: list[float] = []

    for utt in tqdm(utt_ids, desc="spectrogram_heatmap"):
        audio_path = _resolve_audio_path(args.input_dir, utt)
        if audio_path is None:
            files_skipped += 1
            log_error("spectrogram_heatmap", f"missing audio for {utt}")
            continue
        try:
            waveform, sr = load_audio(audio_path)
            score, label = _original_metadata(score_df, utt, score_col)
            mel, attribution, mel_db = _occlusion_attribution(
                model,
                waveform.numpy(),
                sr,
                score,
                args.block_mel,
                args.block_time,
            )
            out_path = args.output_dir / f"{utt}_heatmap.png"
            _save_figure(out_path, mel_db, attribution, sr, 160, score, label, utt, args.block_mel, args.block_time)
            output_paths.append(str(out_path))
            files_processed += 1
            score_accum.append(score)
        except Exception as exc:
            files_failed += 1
            log_error("spectrogram_heatmap", f"{utt}: {exc}")

    write_manifest(args.output_dir, output_paths)
    write_summary(
        "spectrogram_heatmap",
        args.output_dir,
        files_processed,
        files_failed,
        files_skipped,
        float(np.mean(score_accum)) if score_accum else 0.0,
        time.time() - start,
    )


if __name__ == "__main__":
    main()
