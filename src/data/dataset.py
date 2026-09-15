from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


class CMManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        sample_rate: int = 16000,
        duration_sec: float = 4.0,
        training: bool = False,
        augment: bool = False,
        trim_silence: bool = False,
        pre_emphasis: bool = False,
        pre_emphasis_coef: float = 0.97,
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_sec)
        self.training = training
        self.augment = bool(augment)
        self.trim_silence = trim_silence
        self.pre_emphasis = pre_emphasis
        self.pre_emphasis_coef = pre_emphasis_coef

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, rel_path: str) -> torch.Tensor:
        path = self.data_root / rel_path
        try:
            wav, sr = sf.read(path, always_2d=False)
            if wav.ndim == 2:
                wav = np.mean(wav, axis=1)
            wav = torch.tensor(wav, dtype=torch.float32)
        except Exception as sf_err:
            try:
                wav, sr = torchaudio.load(str(path))
                if wav.ndim == 2 and wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze(0).to(dtype=torch.float32)
            except Exception as ta_err:
                try:
                    wav = self._load_with_ffmpeg(path)
                    sr = self.sample_rate
                except Exception as ff_err:
                    raise RuntimeError(f"Failed to load audio file: {path}") from ff_err

        if sr != self.sample_rate:
            wav = F.resample(wav, sr, self.sample_rate)

        if self.pre_emphasis:
            wav = self._pre_emphasis(wav)

        if self.trim_silence:
            wav = self._trim_by_energy(wav)

        if self.training and self.augment:
            wav = self._augment_waveform(wav)

        wav = self._pad_or_crop(wav)
        return wav.unsqueeze(0)

    def _augment_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.numel() == 0:
            return wav

        gain = torch.empty(1).uniform_(0.8, 1.2).item()
        wav = wav * gain

        if torch.rand(1).item() < 0.5:
            signal_power = wav.pow(2).mean().clamp_min(1e-8)
            snr_db = torch.empty(1).uniform_(10.0, 30.0).item()
            noise_power = signal_power / (10.0 ** (snr_db / 10.0))
            noise = torch.randn_like(wav)
            noise = noise / noise.pow(2).mean().clamp_min(1e-8).sqrt()
            wav = wav + noise * noise_power.sqrt()

        if torch.rand(1).item() < 0.25 and wav.numel() > 10:
            max_shift = max(1, int(0.05 * self.sample_rate))
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
            wav = torch.roll(wav, shifts=shift)

        if torch.rand(1).item() < 0.15 and wav.numel() > self.sample_rate // 4:
            seg = int(torch.randint(self.sample_rate // 10, self.sample_rate // 2, (1,)).item())
            seg = min(seg, wav.numel())
            start = int(torch.randint(0, wav.numel() - seg + 1, (1,)).item())
            wav = wav.clone()
            wav[start : start + seg] *= 0.5

        return wav

    def _load_with_ffmpeg(self, path: Path) -> torch.Tensor:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg decode failed for {path}: {stderr}")
        wav = np.frombuffer(proc.stdout, dtype=np.float32)
        if wav.size == 0:
            raise RuntimeError(f"ffmpeg produced empty audio for {path}")
        return torch.from_numpy(wav.copy())

    def _pre_emphasis(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.numel() < 2:
            return wav
        out = wav.clone()
        out[1:] = wav[1:] - self.pre_emphasis_coef * wav[:-1]
        return out

    def _trim_by_energy(self, wav: torch.Tensor, frame: int = 320, hop: int = 160, threshold_db: float = -45.0) -> torch.Tensor:
        if wav.numel() < frame:
            return wav
        unfolded = wav.unfold(0, frame, hop)
        rms = torch.sqrt(torch.mean(unfolded * unfolded, dim=1) + 1e-9)
        rms_db = 20.0 * torch.log10(rms + 1e-8)
        idx = torch.where(rms_db > threshold_db)[0]
        if idx.numel() == 0:
            return wav
        start = max(int(idx[0]) * hop, 0)
        end = min(int(idx[-1]) * hop + frame, wav.numel())
        return wav[start:end]

    def _pad_or_crop(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.numel()
        if n == self.num_samples:
            return wav
        if n < self.num_samples:
            pad = self.num_samples - n
            return torch.nn.functional.pad(wav, (0, pad))

        if self.training:
            start = torch.randint(0, n - self.num_samples + 1, (1,)).item()
        else:
            start = (n - self.num_samples) // 2
        return wav[start : start + self.num_samples]

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        wav = self._load_audio(row["relative_path"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return wav, label, row["relative_path"]


def class_counts_from_manifest(manifest_path: str) -> dict:
    df = pd.read_csv(manifest_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in manifest: {manifest_path}")

    labels = df["label"].astype(int)
    bonafide = int((labels == 0).sum())
    spoof = int((labels == 1).sum())
    return {
        "total": int(len(df)),
        "bonafide": bonafide,
        "spoof": spoof,
    }


def balanced_sampler_from_manifest(manifest_path: str):
    df = pd.read_csv(manifest_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in manifest: {manifest_path}")

    labels = df["label"].astype(int)
    counts = labels.value_counts().to_dict()
    sample_weights = labels.map(lambda label: 1.0 / max(1, counts.get(int(label), 1))).astype(float).to_numpy()
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    class_weights = {str(k): float(1.0 / max(1, v)) for k, v in counts.items()}
    return sampler, class_weights
