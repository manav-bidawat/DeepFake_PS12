"""RawNet-style waveform model for Task 1."""

from __future__ import annotations

import torch
from torch import nn


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.net(self.pool(x))
        return x * scale


class RawResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcite1d(out_channels)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x = x + residual
        return self.act2(x)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(x), dim=-1)
        mean = torch.sum(weights * x, dim=-1)
        centered = x - mean.unsqueeze(-1)
        var = torch.sum(weights * centered.pow(2), dim=-1).clamp_min(1e-8)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)


class RawNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=3, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder = nn.Sequential(
            RawResidualBlock(64, 64, stride=1),
            RawResidualBlock(64, 128, stride=2),
            RawResidualBlock(128, 128, stride=1),
            RawResidualBlock(128, 256, stride=2),
            RawResidualBlock(256, 256, stride=1),
        )
        self.pool = AttentiveStatsPooling(256)
        self.project = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if wav.dim() != 3:
            raise ValueError(f"Expected waveform shape [B, 1, T] or [B, T], got {tuple(wav.shape)}")

        x = self.stem(wav)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.project(x)
        return self.classifier(x).squeeze(-1)