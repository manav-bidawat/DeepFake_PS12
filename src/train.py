import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CMManifestDataset
from src.data.dataset import balanced_sampler_from_manifest
from src.metrics import binary_metrics, eer_and_threshold
from src.models import FocalLoss, build_model, get_lfcc_config
from src.utils import ensure_dir, extract_logits, save_json, set_seed, trainable_params
from src.data.dataset import class_counts_from_manifest


def run_epoch(model, loader, optimizer, loss_fn, device, use_amp: bool = False, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n = 0
    all_scores = []
    all_labels = []
    amp_enabled = bool(use_amp and str(device).startswith("cuda"))

    for wav, label, _ in tqdm(loader, leave=False):
        wav = wav.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = extract_logits(model(wav)).reshape(-1)
                loss = loss_fn(logits, label)

            if train_mode:
                optimizer.zero_grad()
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * wav.size(0)
        n += wav.size(0)
        all_scores.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels).astype(int)
    avg_loss = total_loss / max(1, n)
    return avg_loss, y_true, y_score


def main():
    parser = argparse.ArgumentParser(description="Train Task 1 anti-spoof model")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["cnn", "crnn", "specrnet", "rawnet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--pre-emphasis", action="store_true")
    parser.add_argument("--pre-emphasis-coef", type=float, default=0.97)
    parser.add_argument("--augment", action="store_true", help="Enable waveform augmentation during training.")
    parser.add_argument("--balance-data", action="store_true", help="Use class-balanced sampling during training.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    use_raw_wave_model = args.model == "rawnet"
    augment_enabled = bool(args.augment or use_raw_wave_model)
    balance_enabled = bool(args.balance_data or use_raw_wave_model)
    amp_enabled = bool(args.amp or use_raw_wave_model) and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_ds = CMManifestDataset(
        manifest_path=args.train_manifest,
        data_root=args.data_root,
        duration_sec=args.duration_sec,
        training=True,
        augment=augment_enabled,
        trim_silence=args.trim_silence,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coef=args.pre_emphasis_coef,
    )
    val_ds = CMManifestDataset(
        manifest_path=args.val_manifest,
        data_root=args.data_root,
        duration_sec=args.duration_sec,
        training=False,
        augment=False,
        trim_silence=args.trim_silence,
        pre_emphasis=args.pre_emphasis,
        pre_emphasis_coef=args.pre_emphasis_coef,
    )

    train_counts_before = class_counts_from_manifest(args.train_manifest)
    train_counts_after = dict(train_counts_before)
    class_weights = None
    train_sampler = None
    if balance_enabled:
        train_sampler, class_weights = balanced_sampler_from_manifest(args.train_manifest)
    aug_summary = {
        "applied": augment_enabled,
        "policy": (
            "Raw-waveform augmentation: random gain, shift, noise injection, and occasional segment attenuation."
            if augment_enabled
            else "No data augmentation used."
        ),
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    model = build_model(args.model).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.model == "specrnet":
        loss_fn = FocalLoss(gamma=2.0)
    elif use_raw_wave_model:
        pos_weight_value = train_counts_before["bonafide"] / max(1, train_counts_before["spoof"])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=args.device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    normalization_note = (
        "Per-sample mean/std normalization in LFCC frontend"
        if args.model == "specrnet"
        else "Raw waveform input with augmentation and class-balanced sampling"
        if use_raw_wave_model
        else "Per-sample mean/std normalization in Mel frontend"
    )
    feature_extractor = (
        {
            "name": "LFCC",
            "config": get_lfcc_config(),
        }
        if args.model == "specrnet"
        else {
            "name": "RawWaveform",
            "config": {
                "sample_rate": 16000,
                "duration_sec": args.duration_sec,
                "augmentation": augment_enabled,
                "class_balancing": balance_enabled,
            },
        }
        if use_raw_wave_model
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

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_eer": [],
        "val_threshold": [],
    }

    best_eer = 1.0
    best = None
    start_epoch = 1

    if args.resume_from:
        resume_path = Path(args.resume_from)
        ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        history = ckpt.get("history", history)
        best = ckpt.get("best", best)

        if best is not None and "val_eer" in best:
            best_eer = float(best["val_eer"])
        else:
            best_eer = float(ckpt.get("best_eer", best_eer))

        last_epoch = int(ckpt.get("epoch", ckpt.get("best", {}).get("epoch", 0)))
        start_epoch = last_epoch + 1

        print(
            f"Resuming from {resume_path} | last_epoch={last_epoch} "
            f"start_epoch={start_epoch} best_eer={best_eer:.4f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, y_true_train, y_score_train = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            args.device,
            use_amp=amp_enabled,
            scaler=scaler,
        )
        val_loss, y_true, y_score = run_epoch(
            model,
            val_loader,
            None,
            loss_fn,
            args.device,
            use_amp=amp_enabled,
            scaler=None,
        )
        val_eer, threshold, _, _ = eer_and_threshold(y_true, y_score)
        y_pred = (y_score >= threshold).astype(int)
        cls = binary_metrics(y_true, y_pred)
        val_acc = float((y_pred == y_true).mean())
        y_pred_train = (y_score_train >= threshold).astype(int)
        train_acc = float((y_pred_train == y_true_train).mean())

        scheduler.step()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_eer"].append(float(val_eer))
        history["val_threshold"].append(float(threshold))

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} val_eer={val_eer:.4f} thr={threshold:.4f}"
        )

        epoch_ckpt = {
            "model_name": args.model,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_eer": float(best_eer),
            "best": best,
            "train_args": vars(args),
            "num_trainable_params": trainable_params(model),
        }

        torch.save(epoch_ckpt, str(Path(args.output_dir) / "last.pt"))

        if val_eer < best_eer:
            best_eer = val_eer
            best = {
                "epoch": epoch,
                "val_eer": float(val_eer),
                "threshold": float(threshold),
                "val_metrics": cls,
            }
            epoch_ckpt["best_eer"] = float(best_eer)
            epoch_ckpt["best"] = best
            torch.save(
                epoch_ckpt,
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
            "augmentation": aug_summary,
            "preprocessing": {
                "pre_emphasis": bool(args.pre_emphasis),
                "pre_emphasis_coef": float(args.pre_emphasis_coef),
                "trim_silence": bool(args.trim_silence),
                "feature_extractor": feature_extractor,
                "spectrogram_normalization": normalization_note,
            },
            "class_imbalance": {
                "enabled": bool(balance_enabled),
                "strategy": "WeightedRandomSampler + positive-class weighting" if balance_enabled else "None",
                "class_weights": class_weights,
            },
        },
    )

    x = np.arange(1, len(history["train_loss"]) + 1)
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
    plt.xlabel("Epoch")
    plt.ylabel("EER")
    plt.title("Validation EER")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(args.output_dir) / "val_eer_curve.png"), dpi=160)
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
