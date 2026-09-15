"""Model factory for Task 1."""

from __future__ import annotations

from torch import nn

from src.models.cnn_baseline import CNNClassifier, CRNNClassifier
from src.models.rawnet import RawNet
from src.models.spec_rnet import SpecRNet


def build_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "cnn":
        return CNNClassifier()
    if model_name == "crnn":
        return CRNNClassifier()
    if model_name == "rawnet":
        return RawNet()
    if model_name in {"audiomamba", "audio_mamba", "audio-mamba"}:
        raise ValueError(
            "AudioMamba was referenced in earlier experiments but its implementation "
            "is not present in the recovered project artifacts."
        )
    if model_name in {"specrnet", "spec_rnet", "spec-rnet"}:
        return SpecRNet()
    raise ValueError(f"Unsupported model: {model_name}")
