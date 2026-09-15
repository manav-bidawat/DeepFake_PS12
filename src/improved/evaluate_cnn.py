from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, det_curve, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CMManifestDataset
from src.metrics import binary_metrics, eer_and_threshold
from src.models.cnn_improved import ImprovedCNNClassifier
from src.utils import ensure_dir, extract_logits, save_json


@torch.no_grad()
def infer_scores(model, loader, device: str):
    model.eval()
    ys = []
    ss = []
    for wav, label, _ in tqdm(loader, leave=False):
        wav = wav.to(device, non_blocking=True)
        logits = extract_logits(model(wav)).reshape(-1)
        score = torch.sigmoid(logits).cpu().numpy()
        ss.append(score)
        ys.append(label.numpy())
    return np.concatenate(ys).astype(int), np.concatenate(ss)


def _make_loader(dataset, batch_size: int, num_workers: int, device: str):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate the improved Task 1 CNN checkpoint")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--calibration-manifest", type=str, default=None, help="Optional target-domain dev manifest for threshold calibration.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--pre-emphasis", action="store_true")
    parser.add_argument("--pre-emphasis-coef", type=float, default=0.97)
    parser.add_argument("--threshold", type=float, default=None, help="If missing, uses calibration threshold or checkpoint best threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers is not None else (os.cpu_count() or 1)
    ensure_dir(args.output_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_mode = ckpt.get("train_args", {}).get("feature_mode", "mel_mfcc")
    model = ImprovedCNNClassifier(specaugment=False, feature_mode=feature_mode).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])

    threshold = args.threshold

    if threshold is None and args.calibration_manifest is not None:
        calib_ds = CMManifestDataset(
            manifest_path=args.calibration_manifest,
            data_root=args.data_root,
            duration_sec=args.duration_sec,
            training=False,
            trim_silence=args.trim_silence,
            pre_emphasis=args.pre_emphasis,
            pre_emphasis_coef=args.pre_emphasis_coef,
        )
        calib_loader = _make_loader(calib_ds, args.batch_size, num_workers, args.device)
        y_true_cal, y_score_cal = infer_scores(model, calib_loader, args.device)
        calib_eer, calib_threshold, _, _ = eer_and_threshold(y_true_cal, y_score_cal)
        threshold = float(calib_threshold)
        save_json(
            str(Path(args.output_dir) / "calibration.json"),
            {
                "manifest": args.calibration_manifest,
                "eer": float(calib_eer),
                "threshold": float(calib_threshold),
                "n_samples": int(len(y_true_cal)),
            },
        )

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

    loader = _make_loader(ds, args.batch_size, num_workers, args.device)

    y_true, y_score = infer_scores(model, loader, args.device)
    y_pred = (y_score >= threshold).astype(int)

    eer, _, fpr, tpr = eer_and_threshold(y_true, y_score)
    cls = binary_metrics(y_true, y_pred)
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "threshold_used": float(threshold),
        "eer": float(eer),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "n_samples": int(len(y_true)),
        "n_samples_skipped": 0,
        "preprocessing": {
            "pre_emphasis": bool(args.pre_emphasis),
            "pre_emphasis_coef": float(args.pre_emphasis_coef),
            "trim_silence": bool(args.trim_silence),
            "feature_extractor": {
                "name": "MelSpectrogram+MFCC" if feature_mode == "mel_mfcc" else ("MelSpectrogram" if feature_mode == "mel" else "MFCC"),
                "config": {
                    "sample_rate": 16000,
                    "n_fft": 512,
                    "hop_length": 160,
                    "n_mels": 80,
                    "n_mfcc": 80,
                },
            },
            "spectrogram_normalization": "Per-sample mean/std normalization in Mel/MFCC frontends",
        },
        **cls,
    }
    save_json(str(Path(args.output_dir) / "metrics.json"), metrics)

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
