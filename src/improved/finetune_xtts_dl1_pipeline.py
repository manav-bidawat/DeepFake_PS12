from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from src.improved.clone_dl1_batch import DEFAULT_TEXTS


def build_speaker_dataset(dl1_dir: Path, out_dir: Path, indices: List[int], speaker_name: str) -> Tuple[Path, Path, List[Path]]:
    ds_dir = out_dir / "dataset"
    wav_dir = ds_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    copied_refs: List[Path] = []
    for idx in indices:
        src = dl1_dir / f"{idx}.wav"
        if not src.exists():
            raise FileNotFoundError(f"Missing source wav: {src}")
        dst_name = f"{idx:03d}.wav"
        dst = wav_dir / dst_name
        shutil.copy2(src, dst)
        copied_refs.append(dst)
        rows.append(
            {
                "audio_file": f"wavs/{dst_name}",
                "text": DEFAULT_TEXTS[idx - 1],
                "speaker_name": speaker_name,
            }
        )

    # Keep at least 2 eval samples.
    eval_n = max(2, int(len(rows) * 0.15))
    eval_rows = rows[-eval_n:]
    train_rows = rows[:-eval_n]
    if not train_rows:
        raise RuntimeError("Insufficient rows after split for training")

    train_csv = ds_dir / "metadata_train.csv"
    eval_csv = ds_dir / "metadata_eval.csv"

    with train_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
        w.writeheader()
        w.writerows(train_rows)

    with eval_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
        w.writeheader()
        w.writerows(eval_rows)

    return train_csv, eval_csv, copied_refs


def load_xtts_model(ckpt: Path, config_path: Path, vocab_path: Path, device: str) -> Xtts:
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(ckpt), vocab_path=str(vocab_path), use_deepspeed=False)
    if device == "cuda" and torch.cuda.is_available():
        model.cuda()
    return model


@torch.no_grad()
def generate_for_speaker(
    model: Xtts,
    speaker_refs: List[Path],
    out_dir: Path,
    speaker_tag: str,
    language: str = "en",
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[str(p) for p in speaker_refs],
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )

    generated: List[Path] = []
    for i, text in enumerate(DEFAULT_TEXTS, start=1):
        out = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=model.config.temperature,
            length_penalty=model.config.length_penalty,
            repetition_penalty=model.config.repetition_penalty,
            top_k=model.config.top_k,
            top_p=model.config.top_p,
        )
        wav = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = out_dir / f"{speaker_tag}_{i:02d}.wav"
        torchaudio.save(str(out_path), wav.cpu(), 24000)
        generated.append(out_path)
    return generated


def write_generated_manifest(data_root: Path, generated_files: List[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "scenario", "split", "label"])
        w.writeheader()
        for p in generated_files:
            rel = p.resolve().relative_to(data_root.resolve())
            w.writerow({"relative_path": str(rel), "scenario": "PA", "split": "eval", "label": ""})


def run_pipeline(
    data_root: Path,
    manifest: Path,
    rir_dir: Path,
    checkpoint: Path,
    processed_root: Path,
    run_name: str,
    device: str,
) -> None:
    cmd = [
        "python",
        "-m",
        "src.improved.tts_postprocess_pipeline",
        "--data-root",
        str(data_root),
        "--manifests",
        str(manifest),
        "--include-scenarios",
        "PA",
        "--rir-dir",
        str(rir_dir),
        "--processed-root",
        str(processed_root),
        "--run-name",
        run_name,
        "--checkpoint",
        str(checkpoint),
        "--device",
        device,
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Pipeline run failed with exit code {proc.returncode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune XTTS for DL1 speaker1/speaker2, generate, then run postprocess chain")
    p.add_argument("--data-root", type=Path, default=Path("."))
    p.add_argument("--dl1-dir", type=Path, default=Path("DL1"))
    p.add_argument("--work-dir", type=Path, default=Path("experiments/dl1_xtts_ft"))
    p.add_argument("--generated-dir", type=Path, default=Path("generated/dl1_xtts_ft"))
    p.add_argument("--manifests-dir", type=Path, default=Path("data/manifests"))
    p.add_argument("--rir-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--processed-root", type=Path, default=Path("processed"))
    p.add_argument("--run-name", type=str, default="redteam_dl1_xtts_ft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--language", type=str, default="en")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_root = args.data_root.resolve()
    dl1_dir = (data_root / args.dl1_dir).resolve()
    work_dir = (data_root / args.work_dir).resolve()
    generated_dir = (data_root / args.generated_dir).resolve()

    spk1_dir = work_dir / "speaker1"
    spk2_dir = work_dir / "speaker2"

    train1, eval1, refs1 = build_speaker_dataset(dl1_dir, spk1_dir, list(range(1, 26)), "speaker1")
    train2, eval2, refs2 = build_speaker_dataset(dl1_dir, spk2_dir, list(range(26, 51)), "speaker2")

    cfg1, base_ckpt1, vocab1, exp1, _ = train_gpt(
        language=args.language,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_accum,
        train_csv=str(train1),
        eval_csv=str(eval1),
        output_path=str(spk1_dir),
    )
    cfg2, base_ckpt2, vocab2, exp2, _ = train_gpt(
        language=args.language,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_acumm=args.grad_accum,
        train_csv=str(train2),
        eval_csv=str(eval2),
        output_path=str(spk2_dir),
    )

    exp1 = Path(exp1)
    exp2 = Path(exp2)
    ft_ckpt1 = exp1 / "best_model.pth"
    ft_ckpt2 = exp2 / "best_model.pth"

    shutil.copy2(cfg1, exp1 / Path(cfg1).name)
    shutil.copy2(vocab1, exp1 / Path(vocab1).name)
    shutil.copy2(cfg2, exp2 / Path(cfg2).name)
    shutil.copy2(vocab2, exp2 / Path(vocab2).name)

    model1 = load_xtts_model(ft_ckpt1, Path(cfg1), Path(vocab1), args.device)
    model2 = load_xtts_model(ft_ckpt2, Path(cfg2), Path(vocab2), args.device)

    out_spk1 = generated_dir / "speaker1"
    out_spk2 = generated_dir / "speaker2"
    gen1 = generate_for_speaker(model1, refs1, out_spk1, "spk1", language=args.language)
    gen2 = generate_for_speaker(model2, refs2, out_spk2, "spk2", language=args.language)

    all_generated = gen1 + gen2
    manifest_path = (data_root / args.manifests_dir / "dl1_xtts_finetuned_generated.csv").resolve()
    write_generated_manifest(data_root, all_generated, manifest_path)

    run_pipeline(
        data_root=data_root,
        manifest=manifest_path,
        rir_dir=(data_root / args.rir_dir).resolve(),
        checkpoint=(data_root / args.checkpoint).resolve(),
        processed_root=args.processed_root,
        run_name=args.run_name,
        device=args.device,
    )

    summary = {
        "speaker1_training_dir": str(spk1_dir),
        "speaker2_training_dir": str(spk2_dir),
        "speaker1_finetuned_checkpoint": str(ft_ckpt1),
        "speaker2_finetuned_checkpoint": str(ft_ckpt2),
        "generated_files": len(all_generated),
        "manifest": str(manifest_path),
        "pipeline_run_root": str((data_root / args.processed_root / args.run_name).resolve()),
        "base_checkpoint_speaker1": str(base_ckpt1),
        "base_checkpoint_speaker2": str(base_ckpt2),
    }

    summary_path = generated_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
