import torch
import torch.nn as nn
import torchaudio


class MelFrontend(nn.Module):
    def __init__(self, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 160, n_mels: int = 80):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            f_min=20,
            f_max=7600,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # Input: [B, 1, T], output: [B, 1, n_mels, frames]
        mel = self.spec(wav.squeeze(1))
        mel = self.to_db(mel)
        mel = (mel - mel.mean(dim=(-1, -2), keepdim=True)) / (mel.std(dim=(-1, -2), keepdim=True) + 1e-6)
        return mel.unsqueeze(1)


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.front = MelFrontend()
        self.encoder = nn.Sequential(
            ConvBlock(1, 24),
            ConvBlock(24, 48),
            ConvBlock(48, 96),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(96, 1),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.front(wav)
        x = self.encoder(x)
        return self.head(x).squeeze(-1)


class CRNNClassifier(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.front = MelFrontend()
        self.conv = nn.Sequential(
            ConvBlock(1, 24),
            ConvBlock(24, 48),
        )
        self.gru = nn.GRU(input_size=48 * 20, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.front(wav)
        x = self.conv(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


def build_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "cnn":
        return CNNClassifier()
    if model_name == "crnn":
        return CRNNClassifier()
    raise ValueError(f"Unsupported model: {model_name}")
