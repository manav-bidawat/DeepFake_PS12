from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentation import AugmentedCMManifestDataset, WaveformAugmentationConfig
from src.data.dataset import CMManifestDataset, class_counts_from_manifest
from src.improved.losses import build_loss, make_pos_weight
from src.metrics import binary_metrics, eer_and_threshold
from src.models.cnn_improved import ImprovedCNNClassifier
from src.utils import ensure_dir, extract_logits, save_json, set_seed, trainable_params


def _set_parallelism(num_workers: int) -> None:
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(max(1, cpu_count))
    if num_workers > 0:
        try:
            torch.set_num_interop_threads(max(1, min(cpu_count, num_workers)))
        except RuntimeError:
            pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def _device_type(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"


def run_epoch(model, loader, optimizer, loss_fn, device: str, use_amp: bool = False):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n = 0
    all_scores = []
    all_labels = []
    scaler = getattr(run_epoch, "scaler", None)
    amp_context = torch.autocast(device_type=_device_type(device), enabled=use_amp)

    for wav, label, _ in tqdm(loader, leave=False):
        wav = wav.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode), amp_context:
            logits = extract_logits(model(wav)).reshape(-1)
            loss = loss_fn(logits, label)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * wav.size(0)
        n += wav.size(0)
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels).astype(int)
    avg_loss = total_loss / max(1, n)
    return avg_loss, y_true, y_score


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, device: str):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the improved Task 1 CNN")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--pre-emphasis", action="store_true")
    parser.add_argument("--pre-emphasis-coef", type=float, default=0.97)
    parser.add_argument("--loss", type=str, default="weighted-bce", choices=["bce", "weighted-bce", "pos-weight", "focal"])
    parser.add_argument("--pos-weight", type=float, default=None, help="Positive-class weight for spoof (label=1). If omitted, uses class balance from the training manifest.")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--feature-mode", type=str, default="mel_mfcc", choices=["mel", "mfcc", "mel_mfcc"])
    parser.add_argument("--specaugment", action="store_true", default=True)
    parser.add_argument("--no-specaugment", action="store_false", dest="specaugment")
    parser.add_argument("--specaugment-freq-mask", type=int, default=30)
    parser.add_argument("--specaugment-time-mask", type=int, default=40)
    parser.add_argument("--specaugment-num-freq-masks", type=int, default=2)
    parser.add_argument("--specaugment-num-time-masks", type=int, default=2)
    parser.add_argument("--specaugment-probability", type=float, default=0.7)
    parser.add_argument("--wave-augment", action="store_true", default=True)
    parser.add_argument("--no-wave-augment", action="store_false", dest="wave_augment")
    parser.add_argument("--aug-noise-prob", type=float, default=0.5)
    parser.add_argument("--aug-noise-snr-min", type=float, default=10.0)
    parser.add_argument("--aug-noise-snr-max", type=float, default=30.0)
    parser.add_argument("--aug-rir-prob", type=float, default=0.25)
    parser.add_argument("--aug-rir-length-ms", type=float, default=120.0)
    parser.add_argument("--aug-codec-prob", type=float, default=0.25)
    parser.add_argument("--aug-speed-prob", type=float, default=0.35)
    parser.add_argument("--aug-speed-min", type=float, default=0.9)
    parser.add_argument("--aug-speed-max", type=float, default=1.1)
    parser.add_argument("--aug-gain-min", type=float, default=0.8)
    parser.add_argument("--aug-gain-max", type=float, default=1.2)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers is not None else (os.cpu_count() or 1)
    set_seed(args.seed)
    _set_parallelism(num_workers)
    ensure_dir(args.output_dir)

    waveform_cfg = WaveformAugmentationConfig(
        enabled=bool(args.wave_augment),
        gain_min=args.aug_gain_min,
        gain_max=args.aug_gain_max,
        noise_probability=args.aug_noise_prob,
        noise_snr_min=args.aug_noise_snr_min,
        noise_snr_max=args.aug_noise_snr_max,
        rir_probability=args.aug_rir_prob,
        rir_length_ms=args.aug_rir_length_ms,
        codec_probability=args.aug_codec_prob,
        speed_probability=args.aug_speed_prob,
        speed_min=args.aug_speed_min,
        speed_max=args.aug_speed_max,
    )

    if args.wave_augment:
        train_ds = AugmentedCMManifestDataset(
            manifest_path=args.train_manifest,
            data_root=args.data_root,
            duration_sec=args.duration_sec,
            training=True,
            trim_silence=args.trim_silence,
            pre_emphasis=args.pre_emphasis,
            pre_emphasis_coef=args.pre_emphasis_coef,
            waveform_cfg=waveform_cfg,
        )
    else:
        train_ds = CMManifestDataset(
            manifest_path=args.train_manifest,
            data_root=args.data_root,
            duration_sec=args.duration_sec,
            training=True,
            trim_silence=args.trim_silence,
            pre_emphasis=args.pre_emphasis,
            pre_emphasis_coef=args.pre_emphasis_coef,
        )

    val_ds = CMManifestDataset(
        manifest_path=args.val_manifest,
        data_root=args.data_root,
        duration_sec=args.duration_sec,
        training=False,
        trim_silence=args.trim_silence,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coef=args.pre_emphasis_coef,
    )

    train_counts_before = class_counts_from_manifest(args.train_manifest)
    train_counts_after = dict(train_counts_before)
    auto_pos_weight = make_pos_weight(train_counts_before["bonafide"], train_counts_before["spoof"])
    pos_weight = args.pos_weight if args.pos_weight is not None else auto_pos_weight

    train_loader = _make_loader(train_ds, args.batch_size, num_workers, shuffle=True, device=args.device)
    val_loader = _make_loader(val_ds, args.batch_size, num_workers, shuffle=False, device=args.device)

    model = ImprovedCNNClassifier(
        specaugment=bool(args.specaugment),
        feature_mode=args.feature_mode,
        specaugment_freq_mask=args.specaugment_freq_mask,
        specaugment_time_mask=args.specaugment_time_mask,
        specaugment_num_freq_masks=args.specaugment_num_freq_masks,
        specaugment_num_time_masks=args.specaugment_num_time_masks,
        specaugment_probability=args.specaugment_probability,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = build_loss(args.loss, pos_weight=pos_weight, gamma=args.focal_gamma, device=args.device)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and args.device.startswith("cuda")))
    run_epoch.scaler = scaler

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
        "val_eer": [],
        "val_threshold": [],
    }

    best_eer = 1.0
    best = None

    for epoch in range(1, args.epochs + 1):
        train_loss, y_true_train, y_score_train = run_epoch(model, train_loader, optimizer, loss_fn, args.device, use_amp=bool(args.amp))
        val_loss, y_true, y_score = run_epoch(model, val_loader, None, loss_fn, args.device, use_amp=False)
        val_eer, threshold, _, _ = eer_and_threshold(y_true, y_score)
        y_pred = (y_score >= threshold).astype(int)
        cls = binary_metrics(y_true, y_pred)
        val_acc = float((y_pred == y_true).mean())
        val_balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        val_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        y_pred_train = (y_score_train >= threshold).astype(int)
        train_acc = float((y_pred_train == y_true_train).mean())

        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_balanced_accuracy"].append(val_balanced_accuracy)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_eer"].append(float(val_eer))
        history["val_threshold"].append(float(threshold))

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"val_bal_acc={val_balanced_accuracy:.4f} val_macro_f1={val_macro_f1:.4f} "
            f"val_eer={val_eer:.4f} thr={threshold:.4f}"
        )

        if val_eer < best_eer:
            best_eer = val_eer
            best = {
                "epoch": epoch,
                "val_eer": float(val_eer),
                "threshold": float(threshold),
                "val_metrics": {
                    **cls,
                    "balanced_accuracy": val_balanced_accuracy,
                    "macro_f1": val_macro_f1,
                    "accuracy": val_acc,
                },
            }
            torch.save(
                {
                    "model_name": "cnn_improved",
                    "model_state_dict": model.state_dict(),
                    "best": best,
                    "train_args": vars(args),
                    "num_trainable_params": trainable_params(model),
                },
                str(Path(args.output_dir) / "best.pt"),
            )

    save_json(str(Path(args.output_dir) / "history.json"), history)
    save_json(str(Path(args.output_dir) / "best_summary.json"), best)
    save_json(
        str(Path(args.output_dir) / "data_summary.json"),
        {
            "train_manifest": args.train_manifest,
            "class_counts_before_augmentation": train_counts_before,
            "class_counts_after_augmentation": train_counts_after,
            "augmentation": {
                "waveform": {
                    "applied": bool(args.wave_augment),
                    "config": waveform_cfg.__dict__,
                },
                "specaugment": {
                    "applied": bool(args.specaugment),
                    "freq_mask": args.specaugment_freq_mask,
                    "time_mask": args.specaugment_time_mask,
                    "num_freq_masks": args.specaugment_num_freq_masks,
                    "num_time_masks": args.specaugment_num_time_masks,
                    "probability": args.specaugment_probability,
                },
            },
            "loss": {
                "name": args.loss,
                "pos_weight": float(pos_weight),
                "focal_gamma": float(args.focal_gamma),
            },
            "feature_mode": args.feature_mode,
            "preprocessing": {
                "pre_emphasis": bool(args.pre_emphasis),
                "pre_emphasis_coef": float(args.pre_emphasis_coef),
                "trim_silence": bool(args.trim_silence),
                "feature_extractor": {
                    "name": "MelSpectrogram+MFCC" if args.feature_mode == "mel_mfcc" else ("MelSpectrogram" if args.feature_mode == "mel" else "MFCC"),
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
        },
    )

    x = np.arange(1, args.epochs + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(x, history["train_loss"], label="train_loss")
    plt.plot(x, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(args.output_dir) / "loss_curve.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, history["val_eer"], label="val_eer")
    plt.plot(x, history["val_balanced_accuracy"], label="val_balanced_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation EER and Balanced Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(args.output_dir) / "val_quality_curve.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(x, history["train_acc"], label="train_acc")
    plt.plot(x, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(args.output_dir) / "accuracy_curve.png"), dpi=160)
    plt.close()

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
