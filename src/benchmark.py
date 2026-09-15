import argparse
import os
import platform
import time

import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as F
from torch import nn
from thop import profile

from src.models import build_model
from src.utils import extract_logits, save_json, trainable_params


class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return extract_logits(self.model(x)).reshape(-1)


def load_5s_waveform(path: str, target_sr: int = 16000):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = torch.tensor(wav, dtype=torch.float32)
    if sr != target_sr:
        wav = F.resample(wav, sr, target_sr)

    n = target_sr * 5
    if wav.numel() < n:
        wav = torch.nn.functional.pad(wav, (0, n - wav.numel()))
    else:
        wav = wav[:n]
    return wav.unsqueeze(0).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Benchmark model MACs and latency on 5-second input")
    parser.add_argument("--data-root", type=str, default=".")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-path", type=str, default=None)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device).eval()
    wrapper = LogitsOnlyWrapper(model).to(args.device).eval()

    if args.sample_path is None:
        df = pd.read_csv(args.manifest)
        args.sample_path = os.path.join(args.data_root, df.iloc[0]["relative_path"])

    x = load_5s_waveform(args.sample_path).to(args.device)

    with torch.no_grad():
        macs, _ = profile(wrapper, inputs=(x,), verbose=False)

        for _ in range(10):
            _ = wrapper(x)

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            _ = wrapper(x)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0 / args.iters
    model_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
    gpu_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else None
    cpu_name = platform.processor() or "unknown"

    out = {
        "checkpoint": args.checkpoint,
        "model_name": ckpt["model_name"],
        "sample_path": args.sample_path,
        "device": args.device,
        "cpu_model": cpu_name,
        "gpu_model": gpu_name,
        "iters": args.iters,
        "macs": float(macs),
        "num_params": int(trainable_params(model)),
        "latency_ms_mean": float(latency_ms),
        "model_size_mb": float(model_size_mb),
    }
    save_json(args.output_json, out)
    print(out)


if __name__ == "__main__":
    main()
