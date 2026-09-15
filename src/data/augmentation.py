from __future__ import annotations

from dataclasses import dataclass
import random

import torch
import torchaudio.functional as TAF
from torch.utils.data import Dataset

from src.data.dataset import CMManifestDataset


@dataclass
class WaveformAugmentationConfig:
    enabled: bool = False
    gain_min: float = 0.8
    gain_max: float = 1.2
    noise_probability: float = 0.5
    noise_snr_min: float = 10.0
    noise_snr_max: float = 30.0
    rir_probability: float = 0.25
    rir_length_ms: float = 120.0
    codec_probability: float = 0.25
    speed_probability: float = 0.35
    speed_min: float = 0.9
    speed_max: float = 1.1


def _pad_or_crop(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    if wav.numel() == num_samples:
        return wav
    if wav.numel() < num_samples:
        return torch.nn.functional.pad(wav, (0, num_samples - wav.numel()))
    start = (wav.numel() - num_samples) // 2
    return wav[start : start + num_samples]


def random_gain(wav: torch.Tensor, gain_min: float, gain_max: float) -> torch.Tensor:
    gain = random.uniform(gain_min, gain_max)
    return wav * gain


def add_noise_at_snr(wav: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = wav.pow(2).mean().clamp_min(1e-8)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(wav)
    noise = noise / noise.pow(2).mean().clamp_min(1e-8).sqrt()
    return wav + noise * noise_power.sqrt()


def synthetic_rir(wav: torch.Tensor, sample_rate: int, rir_length_ms: float) -> torch.Tensor:
    rir_length = max(16, int(sample_rate * rir_length_ms / 1000.0))
    time = torch.linspace(0.0, 1.0, steps=rir_length, device=wav.device, dtype=wav.dtype)
    decay = random.uniform(3.0, 8.0)
    rir = torch.randn(rir_length, device=wav.device, dtype=wav.dtype) * torch.exp(-decay * time)
    rir[0] += 1.0
    rir = rir / rir.abs().sum().clamp_min(1e-6)
    filtered = torch.nn.functional.conv1d(
        wav.unsqueeze(0).unsqueeze(0),
        rir.flip(0).view(1, 1, -1),
        padding=rir_length - 1,
    ).squeeze(0).squeeze(0)
    return filtered


def codec_compress(wav: torch.Tensor, encoding_channels: int = 256) -> torch.Tensor:
    clipped = wav.clamp(-1.0, 1.0)
    encoded = TAF.mu_law_encoding(clipped, quantization_channels=encoding_channels)
    decoded = TAF.mu_law_decoding(encoded, quantization_channels=encoding_channels)
    return decoded


def speed_perturb(wav: torch.Tensor, sample_rate: int, speed_factor: float) -> torch.Tensor:
    perturbed_rate = max(1000, int(sample_rate * speed_factor))
    return TAF.resample(wav, sample_rate, perturbed_rate)


def apply_waveform_augmentations(
    wav: torch.Tensor,
    sample_rate: int,
    cfg: WaveformAugmentationConfig,
) -> torch.Tensor:
    if not cfg.enabled:
        return wav

    out = wav
    out = random_gain(out, cfg.gain_min, cfg.gain_max)

    if random.random() < cfg.noise_probability:
        snr_db = random.uniform(cfg.noise_snr_min, cfg.noise_snr_max)
        out = add_noise_at_snr(out, snr_db)

    if random.random() < cfg.rir_probability:
        out = synthetic_rir(out, sample_rate=sample_rate, rir_length_ms=cfg.rir_length_ms)

    if random.random() < cfg.codec_probability:
        out = codec_compress(out)

    if random.random() < cfg.speed_probability:
        speed_factor = random.uniform(cfg.speed_min, cfg.speed_max)
        out = speed_perturb(out, sample_rate=sample_rate, speed_factor=speed_factor)

    return out


class AugmentedCMManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        sample_rate: int = 16000,
        duration_sec: float = 4.0,
        training: bool = False,
        trim_silence: bool = False,
        pre_emphasis: bool = False,
        pre_emphasis_coef: float = 0.97,
        waveform_cfg: WaveformAugmentationConfig | None = None,
    ) -> None:
        self.base = CMManifestDataset(
            manifest_path=manifest_path,
            data_root=data_root,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            training=training,
            trim_silence=trim_silence,
            pre_emphasis=pre_emphasis,
            pre_emphasis_coef=pre_emphasis_coef,
        )
        self.sample_rate = sample_rate
        self.num_samples = self.base.num_samples
        self.training = training
        self.waveform_cfg = waveform_cfg or WaveformAugmentationConfig(enabled=False)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        wav, label, rel_path = self.base[index]
        if self.training and self.waveform_cfg.enabled:
            wav = wav.squeeze(0)
            wav = apply_waveform_augmentations(wav, self.sample_rate, self.waveform_cfg)
            wav = _pad_or_crop(wav, self.num_samples)
            wav = wav.unsqueeze(0)
        return wav, label, rel_path
