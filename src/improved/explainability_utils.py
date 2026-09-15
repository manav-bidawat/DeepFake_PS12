from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml

from src.models import build_model
from src.utils import extract_logits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPLAINABILITY_ROOT = PROJECT_ROOT / "explainability"
ERROR_LOG = EXPLAINABILITY_ROOT / "errors.log"
CLEANUP_LOG = EXPLAINABILITY_ROOT / "cleanup.log"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    if path.suffix.lower() in {".yml", ".yaml"}:
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {"raw": loaded}
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {"raw": loaded}
    try:
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {"raw": loaded}
    except Exception:
        return {"raw_text": text}


def _model_name_from_sources(config: dict, checkpoint: dict) -> str:
    for key in ("model_name", "model", "model_type"):
        if key in config and config[key]:
            return str(config[key])
        if key in checkpoint and checkpoint[key]:
            return str(checkpoint[key])
        train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint.get("train_args"), dict) else {}
        if key in train_args and train_args[key]:
            return str(train_args[key])
    return str(config.get("model_name") or checkpoint.get("model_name") or "cnn")


def load_aasist(config_path: str | Path, checkpoint_path: str | Path, device: str | None = None) -> torch.nn.Module:
    config = load_config(config_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_name = _model_name_from_sources(config, checkpoint)
    model = build_model(model_name)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def infer_aasist(model: torch.nn.Module, waveform: torch.Tensor, sr: int, device: str | None = None) -> float:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    target_sr = 16000
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    device = device or next(model.parameters()).device
    logits = extract_logits(model(waveform.to(device))).reshape(-1)
    score = torch.sigmoid(logits)[0].detach().cpu().item()
    return float(np.clip(score, 0.0, 1.0))


def load_audio(path: str | Path, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr


def get_top_n_files(scores_csv: str | Path, n: int) -> list[str]:
    df = pd.read_csv(scores_csv)
    if "utt_id" not in df.columns:
        raise ValueError(f"scores_csv missing utt_id: {scores_csv}")
    score_col = None
    for candidate in ("after_score", "score_after", "post_score"):
        if candidate in df.columns:
            score_col = candidate
            break
    if score_col is None:
        raise ValueError(f"scores_csv missing after_score/score_after column: {scores_csv}")
    df = df.sort_values(score_col, ascending=False)
    return df.head(n)["utt_id"].astype(str).tolist()


def get_score_column(scores_csv: str | Path) -> str:
    df = pd.read_csv(scores_csv, nrows=1)
    for candidate in ("after_score", "score_after", "post_score"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not find after score column in {scores_csv}")


def clean_output_dir(output_dir: str | Path) -> list[str]:
    out = Path(output_dir)
    ensure_dir(out)
    ensure_dir(EXPLAINABILITY_ROOT)
    cleaned: list[str] = []
    seen_hashes: dict[str, Path] = {}

    def _log(message: str) -> None:
        with CLEANUP_LOG.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")

    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size == 0 or path.suffix == ".tmp":
                path.unlink(missing_ok=True)
                cleaned.append(str(path))
                _log(f"deleted {path}")
                continue
            with path.open("rb") as f:
                digest = hashlib.sha1(f.read()).hexdigest()
            if digest in seen_hashes:
                path.unlink(missing_ok=True)
                cleaned.append(str(path))
                _log(f"deleted duplicate {path} (duplicate of {seen_hashes[digest]})")
            else:
                seen_hashes[digest] = path
        except Exception as exc:
            _log(f"cleanup error on {path}: {exc}")
    return cleaned


def log_error(script_name: str, message: str) -> None:
    ensure_dir(EXPLAINABILITY_ROOT)
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {script_name}: {message}\n")


def write_summary(script_name: str, output_dir: str | Path, files_processed: int, files_failed: int, files_skipped: int, avg_aasist_score: float, runtime_seconds: float) -> Path:
    out = Path(output_dir)
    ensure_dir(out)
    summary = {
        "files_processed": int(files_processed),
        "files_failed": int(files_failed),
        "files_skipped": int(files_skipped),
        "avg_aasist_score": float(avg_aasist_score),
        "runtime_seconds": float(runtime_seconds),
        "output_dir": str(out),
        "timestamp": datetime.now().isoformat(),
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def write_manifest(output_dir: str | Path, output_paths: Iterable[str | Path]) -> Path:
    out = Path(output_dir)
    ensure_dir(out)
    path = out / "manifest.txt"
    lines = [str(Path(p)) for p in output_paths]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
