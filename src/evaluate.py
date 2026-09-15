import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import det_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CMManifestDataset
from src.metrics import binary_metrics, eer_and_threshold
from src.models import build_model, get_lfcc_config
from src.utils import ensure_dir, extract_logits, save_json


@torch.no_grad()
def infer_scores(model, loader, device):
    model.eval()
    ys = []
    ss = []
    for wav, label, _ in tqdm(loader, leave=False):
        wav = wav.to(device)
        logits = extract_logits(model(wav)).reshape(-1)
        score = torch.sigmoid(logits).cpu().numpy()
        ss.append(score)
        ys.append(label.numpy())
    return np.concatenate(ys).astype(int), np.concatenate(ss)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task 1 anti-spoof checkpoint")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--pre-emphasis", action="store_true")
    parser.add_argument("--pre-emphasis-coef", type=float, default=0.97)
    parser.add_argument("--threshold", type=float, default=None, help="If missing, uses checkpoint best threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"]).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    threshold = args.threshold
    if threshold is None:
        threshold = float(ckpt["best"]["threshold"])

    ds = CMManifestDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        duration_sec=args.duration_sec,
        training=False,
        trim_silence=args.trim_silence,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coef=args.pre_emphasis_coef,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    y_true, y_score = infer_scores(model, loader, args.device)
    y_pred = (y_score >= threshold).astype(int)

    eer, _, fpr, tpr = eer_and_threshold(y_true, y_score)
    cls = binary_metrics(y_true, y_pred)
    normalization_note = (
        "Per-sample mean/std normalization in LFCC frontend"
        if ckpt.get("model_name") in {"specrnet", "spec_rnet", "spec-rnet"}
        else "Raw waveform input"
        if ckpt.get("model_name") in {"rawnet", "audiomamba", "audio_mamba", "audio-mamba"}
        else "Per-sample mean/std normalization in Mel frontend"
    )

    feature_extractor = (
        {
            "name": "LFCC",
            "config": get_lfcc_config(),
        }
        if ckpt.get("model_name") in {"specrnet", "spec_rnet", "spec-rnet"}
        else {
            "name": "RawWaveform",
            "config": {
                "sample_rate": 16000,
                "duration_sec": args.duration_sec,
            },
        }
        if ckpt.get("model_name") in {"rawnet", "audiomamba", "audio_mamba", "audio-mamba"}
        else {
            "name": "MelSpectrogram",
            "config": {
                "sample_rate": 16000,
                "n_fft": 512,
                "hop_length": 160,
                "n_mels": 80,
            },
        }
    )

    metrics = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "threshold_used": float(threshold),
        "eer": float(eer),
        "accuracy": float((y_pred == y_true).mean()),
        "n_samples": int(len(y_true)),
        "n_samples_skipped": 0,
        "preprocessing": {
            "pre_emphasis": bool(args.pre_emphasis),
            "pre_emphasis_coef": float(args.pre_emphasis_coef),
            "trim_silence": bool(args.trim_silence),
            "feature_extractor": feature_extractor,
            "spectrogram_normalization": normalization_note,
        },
        **cls,
    }
    save_json(str(Path(args.output_dir) / "metrics.json"), metrics)

    # ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (EER={eer:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(args.output_dir) / "roc.png"), dpi=160)
    plt.close()

    # DET
    fpr_det, fnr_det, _ = det_curve(y_true, y_score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_det, fnr_det, label=f"DET (EER={eer:.4f})")
    plt.scatter([eer], [eer], s=25, label="EER point")
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("DET Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(args.output_dir) / "det.png"), dpi=160)
    plt.close()

    print(metrics)


if __name__ == "__main__":
    main()
