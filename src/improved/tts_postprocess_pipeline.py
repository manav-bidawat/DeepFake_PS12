from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.signal import butter, fftconvolve, sosfiltfilt
from tqdm import tqdm

from src.metrics import eer_and_threshold
from src.models import build_model
from src.utils import extract_logits


@dataclass
class Item:
    utt_id: str
    label: Optional[int]
    relative_path: str
    scenario: str
    split: str
    text: str = ""
    reference_audio: str = ""


@dataclass
class StageFailure:
    utt_id: str
    stage: str
    relative_path: str
    error: str


@dataclass
class ProcessingContext:
    data_root: Path
    run_root: Path
    rir_paths: List[Path]
    noise_min_dbfs: float
    noise_max_dbfs: float
    target_sr: int
    device: str
    rng: random.Random
    rir_cache: Dict[Path, Tuple[np.ndarray, int]] = field(default_factory=dict)


class DacCodec:
    def __init__(self, device: str, bitrate_kbps: int = 8):
        self.device = device
        self.bitrate_kbps = bitrate_kbps
        self._loaded = False
        self._model = None
        self._model_sample_rate = 44100
        self._api_mode = ""
        self._import_error = None

    def _try_load_model(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            import dac  # type: ignore
        except Exception as exc:  # pragma: no cover
            self._import_error = exc
            return

        model = None
        mode = ""

        # Preferred API from descript-audio-codec package.
        try:
            if hasattr(dac, "utils") and hasattr(dac.utils, "download_model"):
                model_path = dac.utils.download_model("44khz")
                model = dac.DAC.load(model_path)
                mode = "download_model"
        except Exception:
            model = None

        try:
            if model is None and hasattr(dac, "utils") and hasattr(dac.utils, "download"):
                model_path = dac.utils.download(model_type="44khz")
                model = dac.DAC.load(model_path)
                mode = "download"
        except Exception:
            model = None

        if model is None:
            self._import_error = RuntimeError(
                "Failed to initialize DAC model. Ensure the `dac` package is installed and model download is available."
            )
            return

        model = model.to(self.device).eval()
        self._model = model
        self._api_mode = mode
        if hasattr(model, "sample_rate"):
            self._model_sample_rate = int(model.sample_rate)

    @torch.no_grad()
    def process(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        self._try_load_model()
        if self._model is None:
            raise RuntimeError(f"DAC unavailable: {self._import_error}")

        x = wav
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.size(0) > 1:
            x = x.mean(dim=0, keepdim=True)

        if sr != self._model_sample_rate:
            x = torchaudio.functional.resample(x, sr, self._model_sample_rate)

        # Expected shape: [B, C, T]
        x = x.unsqueeze(0).to(self.device)

        if hasattr(self._model, "quantizer") and hasattr(self._model.quantizer, "from_bitrate"):
            try:
                n_q = int(self._model.quantizer.from_bitrate(self.bitrate_kbps))
                z = self._model.encode(x, n_quantizers=n_q)
            except Exception:
                z = self._model.encode(x)
        else:
            z = self._model.encode(x)

        # Handle API variants.
        if isinstance(z, (tuple, list)):
            z_latent = z[0]
        else:
            z_latent = z

        y = self._model.decode(z_latent)
        if isinstance(y, (tuple, list)):
            y = y[0]

        y = y.squeeze(0).detach().cpu()
        if y.ndim == 2 and y.size(0) == 1:
            y = y.squeeze(0)

        if self._model_sample_rate != sr:
            if y.ndim == 1:
                y = y.unsqueeze(0)
            y = torchaudio.functional.resample(y, self._model_sample_rate, sr)
            y = y.squeeze(0)
        return y


def _safe_rel_wav_path(rel_path: str) -> Path:
    return Path(rel_path).with_suffix(".wav")


def _read_audio(path: Path) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def _save_audio(path: Path, wav: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav.cpu(), sr, encoding="PCM_S", bits_per_sample=16)


def _load_rir_cached(ctx: ProcessingContext, rir_path: Path) -> Tuple[np.ndarray, int]:
    if rir_path in ctx.rir_cache:
        return ctx.rir_cache[rir_path]
    rir_wav, rir_sr = torchaudio.load(str(rir_path))
    rir = rir_wav.mean(dim=0).cpu().numpy().astype(np.float32)
    peak = float(np.max(np.abs(rir)) + 1e-8)
    rir = rir / peak
    ctx.rir_cache[rir_path] = (rir, int(rir_sr))
    return ctx.rir_cache[rir_path]


def stage1_tts_generation_or_ingest(
    ctx: ProcessingContext,
    item: Item,
    in_path: Path,
    out_path: Path,
    tts_command_template: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not tts_command_template:
        shutil.copy2(in_path, out_path)
        return

    ref_audio = in_path
    if item.reference_audio:
        candidate = (ctx.data_root / item.reference_audio).resolve()
        if candidate.exists():
            ref_audio = candidate

    cmd = tts_command_template.format(
        utt_id=item.utt_id,
        text=item.text,
        input_audio=str(in_path.resolve()),
        reference_audio=str(ref_audio),
        output_wav=str(out_path.resolve()),
    )
    proc = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"TTS command failed for {item.utt_id}: {stderr}")
    if not out_path.exists():
        raise RuntimeError(f"TTS command did not produce output file: {out_path}")


def stage2_rir_convolution(ctx: ProcessingContext, in_path: Path, out_path: Path) -> None:
    wav, sr = _read_audio(in_path)
    x = wav.squeeze(0).cpu().numpy().astype(np.float32)
    rir_path = ctx.rng.choice(ctx.rir_paths)
    rir, rir_sr = _load_rir_cached(ctx, rir_path)
    if rir_sr != sr:
        rir_t = torch.from_numpy(rir).unsqueeze(0)
        rir = torchaudio.functional.resample(rir_t, rir_sr, sr).squeeze(0).numpy().astype(np.float32)

    y = fftconvolve(x, rir, mode="full")[: x.shape[0]]
    in_peak = float(np.max(np.abs(x)) + 1e-8)
    out_peak = float(np.max(np.abs(y)) + 1e-8)
    y = (y / out_peak) * min(in_peak, 0.98)
    _save_audio(out_path, torch.from_numpy(y), sr)


def stage3_dac_wash(dac_codec: DacCodec, in_path: Path, out_path: Path) -> None:
    wav, sr = _read_audio(in_path)
    y = dac_codec.process(wav.squeeze(0), sr)
    _save_audio(out_path, y, sr)


def _make_bandpass_sos(sr: int, low_hz: float = 80.0, high_hz: float = 7800.0):
    nyq = 0.5 * sr
    low = max(1.0, low_hz) / nyq
    high = min(high_hz, nyq - 20.0) / nyq
    if low >= high:
        low = max(1.0, min(low_hz, nyq * 0.1)) / nyq
        high = min(max(low + 0.05, 0.2), 0.95)
    return butter(4, [low, high], btype="bandpass", output="sos")


def stage4_band_limited_noise(ctx: ProcessingContext, in_path: Path, out_path: Path) -> None:
    wav, sr = _read_audio(in_path)
    x = wav.squeeze(0).cpu().numpy().astype(np.float32)

    noise = np.random.randn(x.shape[0]).astype(np.float32)
    sos = _make_bandpass_sos(sr, 80.0, 7800.0)
    noise = sosfiltfilt(sos, noise).astype(np.float32)

    target_dbfs = ctx.rng.uniform(ctx.noise_min_dbfs, ctx.noise_max_dbfs)
    target_rms = float(10.0 ** (target_dbfs / 20.0))
    rms = float(np.sqrt(np.mean(noise * noise) + 1e-12))
    noise = noise * (target_rms / (rms + 1e-12))

    y = np.clip(x + noise, -1.0, 1.0)
    _save_audio(out_path, torch.from_numpy(y), sr)


def stage5_resample_16k_sox(in_path: Path, out_path: Path, target_sr: int) -> None:
    wav, sr = _read_audio(in_path)
    if sr == target_sr:
        _save_audio(out_path, wav, sr)
        return

    try:
        y, out_sr = torchaudio.sox_effects.apply_effects_tensor(
            wav,
            sr,
            effects=[["rate", "-v", str(target_sr)]],
        )
        _save_audio(out_path, y, int(out_sr))
        return
    except Exception:
        # Fallback for environments without SoX bindings.
        y = torchaudio.functional.resample(wav, sr, target_sr)
        _save_audio(out_path, y, int(target_sr))


def _mp3_with_pydub(in_path: Path, out_path: Path, bitrate_k: int) -> bool:
    try:
        from pydub import AudioSegment  # type: ignore
    except Exception:
        return False

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmpf:
        tmp_mp3 = Path(tmpf.name)
    try:
        seg = AudioSegment.from_file(str(in_path))
        seg.export(str(tmp_mp3), format="mp3", bitrate=f"{bitrate_k}k")
        back = AudioSegment.from_file(str(tmp_mp3), format="mp3")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        back.export(str(out_path), format="wav")
        return True
    finally:
        if tmp_mp3.exists():
            tmp_mp3.unlink()


def _mp3_with_ffmpeg(in_path: Path, out_path: Path, bitrate_k: int) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmpf:
        tmp_mp3 = Path(tmpf.name)
    try:
        cmd_enc = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(in_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            f"{bitrate_k}k",
            str(tmp_mp3),
        ]
        cmd_dec = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(tmp_mp3),
            str(out_path),
        ]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        p1 = subprocess.run(cmd_enc, check=False, capture_output=True)
        if p1.returncode != 0:
            err = p1.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg mp3 encode failed: {err}")
        p2 = subprocess.run(cmd_dec, check=False, capture_output=True)
        if p2.returncode != 0:
            err = p2.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg mp3 decode failed: {err}")
    finally:
        if tmp_mp3.exists():
            tmp_mp3.unlink()


def stage6_mp3_passthrough(in_path: Path, out_path: Path, bitrate_k: int = 128) -> None:
    ok = _mp3_with_pydub(in_path, out_path, bitrate_k)
    if not ok:
        _mp3_with_ffmpeg(in_path, out_path, bitrate_k)


@torch.no_grad()
def load_scorer(checkpoint_path: Path, device: str):
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "model_name" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must include model_name and model_state_dict")
    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    threshold = None
    if isinstance(ckpt.get("best"), dict) and "threshold" in ckpt["best"]:
        threshold = float(ckpt["best"]["threshold"])
    return model, threshold


@torch.no_grad()
def score_files(
    model: torch.nn.Module,
    device: str,
    file_paths: Sequence[Path],
    desc: str,
) -> Dict[Path, float]:
    out: Dict[Path, float] = {}
    for path in tqdm(file_paths, desc=desc):
        wav, sr = _read_audio(path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        x = wav.unsqueeze(0).to(device)
        logits = extract_logits(model(x)).reshape(-1)
        score = torch.sigmoid(logits)[0].item()
        out[path] = float(score)
    return out


def parse_items(data_root: Path, manifests: Sequence[Path], include_scenarios: Sequence[str]) -> List[Item]:
    items: List[Item] = []
    include_set = {s.strip().upper() for s in include_scenarios if s.strip()}
    for manifest in manifests:
        df = pd.read_csv(manifest)
        required = {"relative_path"}
        missing = [x for x in required if x not in df.columns]
        if missing:
            raise ValueError(f"Manifest missing columns {missing}: {manifest}")

        for _, row in df.iterrows():
            rel = str(row["relative_path"])
            p = data_root / rel
            if not p.exists():
                continue

            scenario = str(row["scenario"]) if "scenario" in row and pd.notna(row["scenario"]) else "UNK"
            split = str(row["split"]) if "split" in row and pd.notna(row["split"]) else "unspecified"
            if include_set and scenario.upper() not in include_set:
                continue

            label = None
            if "label" in row and pd.notna(row["label"]):
                try:
                    label = int(row["label"])
                except Exception:
                    label = None
            utt = Path(rel).stem
            text = ""
            if "text" in row and pd.notna(row["text"]):
                text = str(row["text"])

            reference_audio = ""
            if "reference_audio" in row and pd.notna(row["reference_audio"]):
                reference_audio = str(row["reference_audio"])

            items.append(
                Item(
                    utt_id=utt,
                    label=label,
                    relative_path=rel,
                    scenario=scenario,
                    split=split,
                    text=text,
                    reference_audio=reference_audio,
                )
            )
    return items


def _sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def clean_workspace(workspace_root: Path, processed_root: Path, logs_dir: Path) -> dict:
    cleaned = {
        "tmp_deleted": [],
        "incomplete_deleted": [],
        "old_logs_deleted": [],
        "duplicate_deleted": [],
        "structure_ok": {},
    }

    for must in [workspace_root / "LA", workspace_root / "PA"]:
        cleaned["structure_ok"][str(must)] = must.exists()
    processed_root.mkdir(parents=True, exist_ok=True)
    cleaned["structure_ok"][str(processed_root)] = processed_root.exists()
    logs_dir.mkdir(parents=True, exist_ok=True)

    for p in workspace_root.rglob("*.tmp"):
        try:
            p.unlink()
            cleaned["tmp_deleted"].append(str(p))
        except Exception:
            pass

    for pattern in ("*.part", "*.incomplete"):
        for p in processed_root.rglob(pattern):
            try:
                p.unlink()
                cleaned["incomplete_deleted"].append(str(p))
            except Exception:
                pass

    logs = sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    for old in logs[1:]:
        try:
            old.unlink()
            cleaned["old_logs_deleted"].append(str(old))
        except Exception:
            pass

    seen_hashes: Dict[str, Path] = {}
    for p in processed_root.rglob("*.wav"):
        try:
            h = _sha1(p)
        except Exception:
            continue
        if h in seen_hashes:
            try:
                p.unlink()
                cleaned["duplicate_deleted"].append(str(p))
            except Exception:
                pass
        else:
            seen_hashes[h] = p

    return cleaned


def _stage_loop(
    stage_name: str,
    stage_dir: Path,
    inputs: Sequence[Item],
    input_file_map: Dict[str, Path],
    failures: List[StageFailure],
    process_fn,
) -> Tuple[List[Item], Dict[str, Path]]:
    outputs: List[Item] = []
    out_map: Dict[str, Path] = {}

    for item in tqdm(inputs, desc=stage_name):
        in_path = input_file_map[item.utt_id]
        out_rel = _safe_rel_wav_path(item.relative_path)
        out_path = stage_dir / out_rel
        try:
            process_fn(item, in_path, out_path)
            outputs.append(item)
            out_map[item.utt_id] = out_path
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            failures.append(StageFailure(item.utt_id, stage_name, item.relative_path, err))
    return outputs, out_map


def _run_optional_tdcf(command_template: str, scores_csv: Path, output_json: Path) -> Tuple[bool, str]:
    cmd = command_template.format(scores_csv=str(scores_csv), output_json=str(output_json))
    p = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        return False, msg
    return True, (p.stdout or "").strip()


def main():
    parser = argparse.ArgumentParser(description="TTS post-processing pipeline for anti-spoof robustness research")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--manifests", type=Path, nargs="+", required=True, help="Input manifests with relative_path and optional label/scenario/split")
    parser.add_argument("--include-scenarios", type=str, default="LA,PA", help="Comma-separated scenario filter, e.g., LA,PA")
    parser.add_argument("--rir-dir", type=Path, required=True, help="Directory containing measured RIR WAV/FLAC files")
    parser.add_argument("--processed-root", type=Path, default=Path("processed"))
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Anti-spoof model checkpoint for score logging")
    parser.add_argument(
        "--tts-command-template",
        type=str,
        default="",
        help=(
            "Optional per-utterance generation command. Available placeholders: "
            "{utt_id}, {text}, {input_audio}, {reference_audio}, {output_wav}"
        ),
    )
    parser.add_argument("--noise-min-dbfs", type=float, default=-82.0)
    parser.add_argument("--noise-max-dbfs", type=float, default=-78.0)
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--mp3-bitrate-kbps", type=int, default=128)
    parser.add_argument("--dac-bitrate-kbps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tdcf-command", type=str, default="", help="Optional command template with {scores_csv} and {output_json}")
    args = parser.parse_args()

    workspace_root = args.data_root.resolve()
    processed_root = (workspace_root / args.processed_root).resolve()
    logs_dir = processed_root / "logs"
    run_root = processed_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"{args.run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    clean_report = clean_workspace(workspace_root, processed_root, logs_dir)
    log("[Cleanup] Workspace hygiene report:")
    log(json.dumps(clean_report, indent=2))

    structure_ok = all(clean_report["structure_ok"].values())
    if not structure_ok:
        missing = [k for k, ok in clean_report["structure_ok"].items() if not ok]
        raise RuntimeError(f"Required folder structure missing: {missing}")

    include_scenarios = [s.strip() for s in args.include_scenarios.split(",") if s.strip()]
    items = parse_items(workspace_root, args.manifests, include_scenarios)
    if not items:
        raise RuntimeError("No valid input files found from provided manifests")

    rir_paths = list(args.rir_dir.rglob("*.wav")) + list(args.rir_dir.rglob("*.flac"))
    if not rir_paths:
        raise RuntimeError(f"No RIR files found in: {args.rir_dir}")

    rng = random.Random(args.seed)
    ctx = ProcessingContext(
        data_root=workspace_root,
        run_root=run_root,
        rir_paths=rir_paths,
        noise_min_dbfs=min(args.noise_min_dbfs, args.noise_max_dbfs),
        noise_max_dbfs=max(args.noise_min_dbfs, args.noise_max_dbfs),
        target_sr=args.target_sr,
        device=args.device,
        rng=rng,
    )

    failures: List[StageFailure] = []
    stage_counts = []

    stage1_dir = run_root / "stage1_tts"
    stage2_dir = run_root / "stage2_rir"
    stage3_dir = run_root / "stage3_dac"
    stage4_dir = run_root / "stage4_noise"
    stage5_dir = run_root / "stage5_resample16k"
    stage6_dir = run_root / "stage6_mp3"

    input_map = {item.utt_id: (workspace_root / item.relative_path) for item in items}

    active, map1 = _stage_loop(
        stage_name="stage1_tts_generation",
        stage_dir=stage1_dir,
        inputs=items,
        input_file_map=input_map,
        failures=failures,
        process_fn=lambda item, a, b: stage1_tts_generation_or_ingest(
            ctx,
            item,
            a,
            b,
            args.tts_command_template,
        ),
    )
    stage_counts.append(("stage1_tts_generation", len(items), len(active)))

    active, map2 = _stage_loop(
        stage_name="stage2_rir_convolution",
        stage_dir=stage2_dir,
        inputs=active,
        input_file_map=map1,
        failures=failures,
        process_fn=lambda _item, a, b: stage2_rir_convolution(ctx, a, b),
    )
    stage_counts.append(("stage2_rir_convolution", len(map1), len(active)))

    dac_codec = DacCodec(device=args.device, bitrate_kbps=args.dac_bitrate_kbps)
    active, map3 = _stage_loop(
        stage_name="stage3_dac_codec_wash",
        stage_dir=stage3_dir,
        inputs=active,
        input_file_map=map2,
        failures=failures,
        process_fn=lambda _item, a, b: stage3_dac_wash(dac_codec, a, b),
    )
    stage_counts.append(("stage3_dac_codec_wash", len(map2), len(active)))

    active, map4 = _stage_loop(
        stage_name="stage4_band_limited_noise",
        stage_dir=stage4_dir,
        inputs=active,
        input_file_map=map3,
        failures=failures,
        process_fn=lambda _item, a, b: stage4_band_limited_noise(ctx, a, b),
    )
    stage_counts.append(("stage4_band_limited_noise", len(map3), len(active)))

    active, map5 = _stage_loop(
        stage_name="stage5_resample_16k_sox",
        stage_dir=stage5_dir,
        inputs=active,
        input_file_map=map4,
        failures=failures,
        process_fn=lambda _item, a, b: stage5_resample_16k_sox(a, b, args.target_sr),
    )
    stage_counts.append(("stage5_resample_16k_sox", len(map4), len(active)))

    active, map6 = _stage_loop(
        stage_name="stage6_mp3_passthrough",
        stage_dir=stage6_dir,
        inputs=active,
        input_file_map=map5,
        failures=failures,
        process_fn=lambda _item, a, b: stage6_mp3_passthrough(a, b, args.mp3_bitrate_kbps),
    )
    stage_counts.append(("stage6_mp3_passthrough", len(map5), len(active)))

    for stage_name, expected, actual in stage_counts:
        msg = f"[StageCount] {stage_name}: input={expected} output={actual}"
        if expected != actual:
            msg += " [MISMATCH]"
        log(msg)

    if not active:
        raise RuntimeError("All files failed before final output stage; see failures.csv")

    model, threshold = load_scorer(args.checkpoint, args.device)

    before_paths = [workspace_root / item.relative_path for item in active]
    after_paths = [map6[item.utt_id] for item in active]

    before_scores = score_files(model, args.device, before_paths, desc="Scoring before")
    after_scores = score_files(model, args.device, after_paths, desc="Scoring after")

    score_rows = []
    labels_before: List[int] = []
    scores_before_vec: List[float] = []
    labels_after: List[int] = []
    scores_after_vec: List[float] = []

    for item in active:
        p_before = workspace_root / item.relative_path
        p_after = map6[item.utt_id]
        s_before = before_scores.get(p_before)
        s_after = after_scores.get(p_after)

        row = {
            "utt_id": item.utt_id,
            "scenario": item.scenario,
            "split": item.split,
            "label": item.label if item.label is not None else "",
            "relative_path_before": item.relative_path,
            "relative_path_after": str(p_after.relative_to(workspace_root)),
            "score_before": "" if s_before is None else s_before,
            "score_after": "" if s_after is None else s_after,
            "score_delta_after_minus_before": "" if (s_before is None or s_after is None) else (s_after - s_before),
        }
        score_rows.append(row)

        if item.label is not None and s_before is not None:
            labels_before.append(int(item.label))
            scores_before_vec.append(float(s_before))
        if item.label is not None and s_after is not None:
            labels_after.append(int(item.label))
            scores_after_vec.append(float(s_after))

    fail_by_utt = {f.utt_id: f for f in failures}
    processed_utts = {it.utt_id for it in active}
    all_utts = {it.utt_id for it in items}
    failed_only = sorted(all_utts - processed_utts)
    for utt in failed_only:
        f = fail_by_utt.get(utt)
        item = next(it for it in items if it.utt_id == utt)
        score_rows.append(
            {
                "utt_id": item.utt_id,
                "scenario": item.scenario,
                "split": item.split,
                "label": item.label if item.label is not None else "",
                "relative_path_before": item.relative_path,
                "relative_path_after": "",
                "score_before": "",
                "score_after": "",
                "score_delta_after_minus_before": "",
                "failed_stage": "" if f is None else f.stage,
                "failed_error": "" if f is None else f.error,
            }
        )

    scores_csv = run_root / "aasist_scores.csv"
    pd.DataFrame(score_rows).to_csv(scores_csv, index=False)

    failures_csv = run_root / "failures.csv"
    with failures_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "stage", "relative_path", "error"])
        for x in failures:
            w.writerow([x.utt_id, x.stage, x.relative_path, x.error])

    summary = {
        "run_name": args.run_name,
        "run_root": str(run_root),
        "input_files": len(items),
        "processed_files": len(active),
        "failed_files": len(all_utts - processed_utts),
        "stage_counts": [
            {"stage": n, "input": exp, "output": out, "match": exp == out}
            for n, exp, out in stage_counts
        ],
        "avg_score_before": float(np.mean(scores_before_vec)) if scores_before_vec else None,
        "avg_score_after": float(np.mean(scores_after_vec)) if scores_after_vec else None,
        "threshold_from_checkpoint": threshold,
        "cleaning": clean_report,
        "metrics": {},
    }

    if labels_before and len(set(labels_before)) == 2:
        eer_before, thr_before, _, _ = eer_and_threshold(np.array(labels_before), np.array(scores_before_vec))
        summary["metrics"]["eer_before"] = float(eer_before)
        summary["metrics"]["eer_before_threshold"] = float(thr_before)

    if labels_after and len(set(labels_after)) == 2:
        eer_after, thr_after, _, _ = eer_and_threshold(np.array(labels_after), np.array(scores_after_vec))
        summary["metrics"]["eer_after"] = float(eer_after)
        summary["metrics"]["eer_after_threshold"] = float(thr_after)

    if args.tdcf_command:
        tdcf_json = run_root / "tdcf.json"
        ok, out = _run_optional_tdcf(args.tdcf_command, scores_csv, tdcf_json)
        summary["metrics"]["tdcf_command_used"] = args.tdcf_command
        summary["metrics"]["tdcf_status"] = "ok" if ok else "failed"
        summary["metrics"]["tdcf_message"] = out
        if tdcf_json.exists():
            try:
                summary["metrics"]["tdcf"] = json.loads(tdcf_json.read_text(encoding="utf-8"))
            except Exception:
                summary["metrics"]["tdcf"] = tdcf_json.read_text(encoding="utf-8", errors="ignore")
    else:
        summary["metrics"]["tdcf"] = "Not computed (no --tdcf-command provided)"

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("[Summary]")
    log(json.dumps(summary, indent=2))

    print("\n=== Final Summary ===")
    print(f"Files processed: {summary['processed_files']} / {summary['input_files']}")
    print(f"Files failed: {summary['failed_files']}")
    print(f"Average AASIST score before: {summary['avg_score_before']}")
    print(f"Average AASIST score after: {summary['avg_score_after']}")
    print(f"Scores CSV: {scores_csv}")
    print(f"Failures CSV: {failures_csv}")
    print(f"Run summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
        raise