from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


@dataclass
class AASISTConfig:
    sample_rate: int = 16000
    n_samples: int = 64600
    pre_emphasis: float = 0.97
    sinc_kernel: int = 1024
    filts: list = field(default_factory=lambda: [70, [70, 32], [32, 32], [32, 64], [64, 64]])
    nb_fc_node: int = 64
    gat_dims: list = field(default_factory=lambda: [64, 32])
    temperatures: list = field(default_factory=lambda: [2.0, 2.0, 100.0, 100.0])


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    @staticmethod
    def to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def __init__(self, out_channels: int, kernel_size: int, sr: int = 16000, min_low_hz: int = 50, min_band_hz: int = 50) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sr
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30.0
        high_hz = sr / 2.0 - (min_low_hz + min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))

        n_lin = torch.linspace(0, kernel_size / 2 - 1, kernel_size // 2)
        self.register_buffer("window_", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size))
        n = (kernel_size - 1) / 2.0
        self.register_buffer("n_", 2 * math.pi * torch.arange(-n, 0).view(1, -1) / sr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        f_l = torch.matmul(low, self.n_)
        f_h = torch.matmul(high, self.n_)
        bp_l = ((torch.sin(f_h) - torch.sin(f_l)) / (self.n_ / 2)) * self.window_
        bp = torch.cat([bp_l, 2 * band.view(-1, 1), torch.flip(bp_l, [1])], 1)
        bp = bp / (2 * band[:, None])
        return F.conv1d(x, bp.view(self.out_channels, 1, self.kernel_size), padding=self.kernel_size // 2)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, first: bool = False) -> None:
        super().__init__()
        self.first = first
        self.lrelu = nn.LeakyReLU(0.3)
        if not first:
            self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = x if self.first else self.lrelu(self.bn1(x))
        out = self.conv1(out)
        out = self.lrelu(self.bn2(out))
        out = self.conv2(out)
        return out + identity


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k_neighbors: int = 4) -> None:
        super().__init__()
        self.k_neighbors = k_neighbors
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        bsz, n_nodes, _ = x.shape
        h = self.proj(x)
        e = self.lrelu(
            self.attn(
                torch.cat(
                    [
                        h.unsqueeze(2).expand(-1, -1, n_nodes, -1),
                        h.unsqueeze(1).expand(-1, n_nodes, -1, -1),
                    ],
                    -1,
                )
            ).squeeze(-1)
        )
        if self.k_neighbors < n_nodes:
            thr = torch.topk(e, min(self.k_neighbors, n_nodes), -1).values[..., -1:]
            e = e.masked_fill(e < thr, float("-inf"))
        a = F.softmax(e, -1)
        out = torch.bmm(a, h)
        return F.elu(out), None


class HSGAL(nn.Module):
    def __init__(self, dim_s: int, dim_t: int, gat_dims: list, k_neighbors: int = 4, temperature: float = 2.0) -> None:
        super().__init__()
        self.gat_s = GraphAttentionLayer(dim_s, gat_dims[0], k_neighbors)
        self.gat_t = GraphAttentionLayer(dim_t, gat_dims[0], k_neighbors)
        self.cross = nn.Linear(gat_dims[0], gat_dims[1])
        self.temp = temperature

    def _attend(self, query_src: torch.Tensor, key_src: torch.Tensor) -> torch.Tensor:
        q = self.cross(query_src.mean(1, keepdim=True))
        k = self.cross(key_src)
        a = F.softmax(torch.bmm(q, k.transpose(1, 2)) / self.temp, -1)
        return torch.bmm(a, k).squeeze(1)

    def forward(self, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        hs, _ = self.gat_s(xs)
        ht, _ = self.gat_t(xt)
        return torch.cat([self._attend(hs, ht), self._attend(ht, hs)], 1)


class AASIST(nn.Module):
    def __init__(self, cfg: AASISTConfig) -> None:
        super().__init__()
        self.sinc = SincConv(70, cfg.sinc_kernel, sr=cfg.sample_rate)
        self.bn0 = nn.BatchNorm1d(70)
        self.pool0 = nn.MaxPool1d(3)
        self.drop0 = nn.Dropout(0.05)

        self.enc_blocks = nn.ModuleList()
        self.enc_pools = nn.ModuleList()
        for i, flt in enumerate(cfg.filts[1:]):
            self.enc_blocks.append(ResBlock(flt[0], flt[1], first=(i == 0)))
            self.enc_pools.append(nn.MaxPool1d(3))

        self.block_last = ResBlock(64, 64)
        self.bn_last = nn.BatchNorm1d(64)
        self.drop_last = nn.Dropout(0.05)

        self.hsgal = HSGAL(64, 64, cfg.gat_dims, k_neighbors=4, temperature=cfg.temperatures[0])
        self.fc1 = nn.Linear(2 * cfg.gat_dims[1], cfg.nb_fc_node)
        self.bn_fc = nn.BatchNorm1d(cfg.nb_fc_node)
        self.fc2 = nn.Linear(cfg.nb_fc_node, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu(self.bn0(torch.abs(self.sinc(x))), 0.3)
        out = self.drop0(self.pool0(out))
        for blk, pool in zip(self.enc_blocks, self.enc_pools):
            out = pool(blk(out))
        out = self.drop_last(F.leaky_relu(self.bn_last(self.block_last(out)), 0.3))
        xs = out.permute(0, 2, 1)
        h = F.leaky_relu(self.bn_fc(self.fc1(self.hsgal(xs, xs))), 0.3)
        return self.fc2(h)


class AASISTClassifier:
    def __init__(self, checkpoint_path: str | Path, threshold: float = 0.420, device: str | None = None) -> None:
        self.cfg = AASISTConfig()
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = float(threshold)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if not self.checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model = AASIST(self.cfg).to(self.device)
        state = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _pre_emphasis_filter(self, wave: np.ndarray) -> np.ndarray:
        if wave.size == 0:
            return wave
        return np.append(wave[0], wave[1:] - self.cfg.pre_emphasis * wave[:-1]).astype(np.float32)

    def _energy_vad(self, wave: np.ndarray, frame_ms: int = 25, hop_ms: int = 10, threshold_db: float = -40.0) -> np.ndarray:
        if wave.size < 8:
            return wave
        sr = self.cfg.sample_rate
        frame_len = int(sr * frame_ms / 1000)
        hop_len = int(sr * hop_ms / 1000)
        if frame_len <= 0 or hop_len <= 0 or wave.size < frame_len:
            return wave
        frames = np.lib.stride_tricks.sliding_window_view(wave, frame_len)[::hop_len]
        if frames.size == 0:
            return wave
        rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-9)
        energy = 20.0 * np.log10(rms + 1e-9)
        voiced = energy > (float(np.max(energy)) + threshold_db)
        if not np.any(voiced):
            return wave
        first = int(np.argmax(voiced)) * hop_len
        last = int((len(voiced) - np.argmax(voiced[::-1])) * hop_len)
        return wave[first:max(last, first + 1)]

    @staticmethod
    def _rms_normalise(wave: np.ndarray, target_db: float = -23.0) -> np.ndarray:
        if wave.size == 0:
            return wave
        rms = np.sqrt(np.mean(wave.astype(np.float32) ** 2) + 1e-9)
        gain = (10.0 ** (target_db / 20.0)) / rms
        return (wave * gain).astype(np.float32)

    def _pad_or_crop(self, wave: np.ndarray) -> np.ndarray:
        n = self.cfg.n_samples
        if wave.size == 0:
            return np.zeros(n, dtype=np.float32)
        if wave.size >= n:
            start = (wave.size - n) // 2
            return wave[start:start + n].astype(np.float32)
        reps = n // max(wave.size, 1) + 1
        return np.tile(wave, reps)[:n].astype(np.float32)

    def _load_wave(self, audio_path: str | Path) -> np.ndarray:
        wav, sr = torchaudio.load(str(audio_path))
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def _preprocess(self, audio_path: str | Path) -> torch.Tensor:
        wave = self._load_wave(audio_path)
        wave = self._pre_emphasis_filter(wave)
        wave = self._energy_vad(wave)
        wave = self._rms_normalise(wave)
        wave = self._pad_or_crop(wave)
        x = torch.from_numpy(wave).unsqueeze(0).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def predict_file(self, audio_path: str | Path) -> Dict[str, Any]:
        x = self._preprocess(audio_path)
        logits = self.model(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        probs = F.softmax(logits, dim=1)[0]
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)

        prob_sum = float(probs.sum().detach().cpu().item())
        if not np.isfinite(prob_sum) or prob_sum <= 0.0:
            spoof_score = 0.5
        else:
            spoof_score = float((probs[1] / probs.sum()).detach().cpu().item())
            if not np.isfinite(spoof_score):
                spoof_score = 0.5

        spoof_score = float(np.clip(spoof_score, 0.0, 1.0))
        is_spoof = spoof_score >= self.threshold
        if is_spoof:
            confidence = (spoof_score - self.threshold) / max(1e-6, 1.0 - self.threshold)
            label = "fake"
        else:
            confidence = (self.threshold - spoof_score) / max(1e-6, self.threshold)
            label = "bonafide"

        if not np.isfinite(confidence):
            confidence = 0.0

        return {
            "label": label,
            "is_spoof": bool(is_spoof),
            "spoof_score": round(spoof_score, 6),
            "threshold": float(self.threshold),
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 6),
        }
