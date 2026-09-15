from __future__ import annotations

import random

import torch
from torch import nn
import torchaudio

from src.models.cnn_baseline import ConvBlock, MelFrontend


class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        p: float = 0.7,
    ) -> None:
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def _mask(self, x: torch.Tensor, dim: int, max_width: int, num_masks: int) -> torch.Tensor:
        out = x.clone()
        for sample_idx in range(out.size(0)):
            for _ in range(num_masks):
                width = int(torch.randint(0, max_width + 1, (1,), device=out.device).item())
                if width == 0:
                    continue
                limit = out.size(dim)
                if width >= limit:
                    continue
                start = int(torch.randint(0, limit - width + 1, (1,), device=out.device).item())
                if dim == 2:
                    out[sample_idx, :, start : start + width, :] = 0
                else:
                    out[sample_idx, :, :, start : start + width] = 0
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x
        x = self._mask(x, dim=2, max_width=self.freq_mask_param, num_masks=self.num_freq_masks)
        x = self._mask(x, dim=3, max_width=self.time_mask_param, num_masks=self.num_time_masks)
        return x


class ImprovedCNNClassifier(nn.Module):
    def __init__(
        self,
        specaugment: bool = True,
        feature_mode: str = "mel_mfcc",
        specaugment_freq_mask: int = 30,
        specaugment_time_mask: int = 40,
        specaugment_num_freq_masks: int = 2,
        specaugment_num_time_masks: int = 2,
        specaugment_probability: float = 0.7,
    ) -> None:
        super().__init__()
        self.feature_mode = feature_mode.lower()
        self.mel_front = MelFrontend()
        self.mfcc_front = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=80,
            melkwargs={
                "n_fft": 512,
                "hop_length": 160,
                "win_length": 512,
                "n_mels": 80,
                "f_min": 20,
                "f_max": 7600,
                "power": 2.0,
            },
        )
        self.specaugment = SpecAugment(
            freq_mask_param=specaugment_freq_mask,
            time_mask_param=specaugment_time_mask,
            num_freq_masks=specaugment_num_freq_masks,
            num_time_masks=specaugment_num_time_masks,
            p=specaugment_probability,
        ) if specaugment else nn.Identity()
        self.encoder = nn.Sequential(
            ConvBlock(2 if self.feature_mode == "mel_mfcc" else 1, 24),
            ConvBlock(24, 48),
            ConvBlock(48, 96),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def _front(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel_front(wav)
        if self.feature_mode == "mel":
            return mel

        mfcc = self.mfcc_front(wav.squeeze(1))
        mfcc = (mfcc - mfcc.mean(dim=(-1, -2), keepdim=True)) / (mfcc.std(dim=(-1, -2), keepdim=True) + 1e-6)
        mfcc = mfcc.unsqueeze(1)

        if self.feature_mode == "mfcc":
            return mfcc

        if self.feature_mode == "mel_mfcc":
            return torch.cat([mel, mfcc], dim=1)

        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self._front(wav)
        x = self.specaugment(x)
        x = self.encoder(x)
        return self.head(x).squeeze(-1)
