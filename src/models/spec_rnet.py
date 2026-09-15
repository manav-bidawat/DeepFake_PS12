"""SpecRNet components for Task 1."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_lfcc_config() -> dict:
    return {
        "sample_rate": 16000,
        "window_length": 400,
        "hop_length": 160,
        "filter_count": 60,
        "num_ceps": 20,
        "n_fft": 512,
        "f_min": 0.0,
        "f_max": 8000.0,
    }


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = (1.0 - pt).pow(self.gamma) * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _create_linear_filterbank(sample_rate: int, n_fft: int, filter_count: int, f_min: float, f_max: float) -> torch.Tensor:
    n_freqs = n_fft // 2 + 1
    freq_bins = torch.linspace(0.0, float(sample_rate) / 2.0, n_freqs)
    edges = torch.linspace(f_min, f_max, filter_count + 2)

    filters = torch.zeros(filter_count, n_freqs)
    for idx in range(filter_count):
        left = edges[idx]
        center = edges[idx + 1]
        right = edges[idx + 2]

        if center > left:
            left_slope = (freq_bins - left) / (center - left)
        else:
            left_slope = torch.zeros_like(freq_bins)
        if right > center:
            right_slope = (right - freq_bins) / (right - center)
        else:
            right_slope = torch.zeros_like(freq_bins)

        filters[idx] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)
    return filters


def _create_dct(num_ceps: int, filter_count: int) -> torch.Tensor:
    n = torch.arange(filter_count, dtype=torch.float32)
    k = torch.arange(num_ceps, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos((np.pi / filter_count) * (n + 0.5) * k)
    basis[0] *= 1.0 / torch.sqrt(torch.tensor(2.0))
    basis *= torch.sqrt(torch.tensor(2.0 / filter_count))
    return basis


class LFCCFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        window_length: int = 400,
        hop_length: int = 160,
        filter_count: int = 60,
        num_ceps: int = 20,
        n_fft: int = 512,
        f_min: float = 0.0,
        f_max: float | None = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.filter_count = filter_count
        self.num_ceps = num_ceps
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = float(f_max if f_max is not None else sample_rate / 2.0)

        self.register_buffer("window", torch.hann_window(window_length), persistent=False)
        self.register_buffer(
            "filterbank",
            _create_linear_filterbank(sample_rate, n_fft, filter_count, self.f_min, self.f_max),
            persistent=False,
        )
        self.register_buffer("dct", _create_dct(num_ceps, filter_count), persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        if wav.dim() != 2:
            raise ValueError(f"Expected waveform shape [B, 1, T] or [B, T], got {tuple(wav.shape)}")

        if wav.size(-1) < self.window_length:
            wav = F.pad(wav, (0, self.window_length - wav.size(-1)))

        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        fbanks = torch.einsum("cf,bft->bct", self.filterbank, power).clamp_min(1e-10)
        cepstra = torch.einsum("kc,bct->bkt", self.dct, torch.log(fbanks))
        cepstra = (cepstra - cepstra.mean(dim=(1, 2), keepdim=True)) / (cepstra.std(dim=(1, 2), keepdim=True, unbiased=False) + 1e-6)
        return cepstra.unsqueeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, negative_slope: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act2 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return self.act2(x)


class SpecRNet(nn.Module):
    def __init__(self):
        super().__init__()
        config = get_lfcc_config()
        self.frontend = LFCCFrontend(**config)
        self.initial_norm = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.SELU(inplace=True),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder = nn.Sequential(
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, 1)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.frontend(wav)
        x = self.initial_norm(x)
        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x)
        embedding = self.embedding(x)
        logits = self.classifier(self.dropout(embedding)).squeeze(-1)
        return logits, embedding
