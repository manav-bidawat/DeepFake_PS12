from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class TTSEngine:
    """XTTS inference wrapper for webapp backend."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.vocab_path = vocab_path
        self.model: Optional[Xtts] = None
        self.output_sr = 24000
        self._load_model()

    def _resolve_paths(self) -> Tuple[Path, Path, Path]:
        model_dir = Path(self.model_path).resolve() if self.model_path else None

        ckpt = Path(self.checkpoint_path).resolve() if self.checkpoint_path else None
        cfg = Path(self.config_path).resolve() if self.config_path else None
        vocab = Path(self.vocab_path).resolve() if self.vocab_path else None

        if model_dir and model_dir.exists():
            if ckpt is None:
                for name in ("best_model.pth", "model.pth", "checkpoint.pth", "model.pth.tar"):
                    p = model_dir / name
                    if p.exists():
                        ckpt = p
                        break
            if cfg is None:
                for name in ("config.json", "config_xtts.json"):
                    p = model_dir / name
                    if p.exists():
                        cfg = p
                        break
            if vocab is None:
                for name in ("vocab.json", "vocab.txt"):
                    p = model_dir / name
                    if p.exists():
                        vocab = p
                        break

        if not ckpt or not ckpt.exists():
            raise RuntimeError("XTTS checkpoint not found. Set --xtts-checkpoint or --model-path")
        if not cfg or not cfg.exists():
            raise RuntimeError("XTTS config not found. Set --xtts-config or --model-path")
        if not vocab or not vocab.exists():
            raise RuntimeError("XTTS vocab not found. Set --xtts-vocab or --model-path")

        return ckpt, cfg, vocab

    def _load_model(self) -> None:
        ckpt, cfg, vocab = self._resolve_paths()
        config = XttsConfig()
        config.load_json(str(cfg))

        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_path=str(ckpt), vocab_path=str(vocab), use_deepspeed=False)
        if self.device == "cuda" and torch.cuda.is_available():
            model.cuda()

        self.model = model
        self.output_sr = int(getattr(config, "output_sample_rate", getattr(config, "sample_rate", 24000)))

    @staticmethod
    def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        sig = inspect.signature(fn)
        accepted = {}
        for k, v in kwargs.items():
            if k in sig.parameters:
                accepted[k] = v
        return accepted

    def _pick_audio_from_result(self, result: Any) -> np.ndarray:
        if result is None:
            raise RuntimeError("XTTS returned no audio")

        if isinstance(result, dict):
            if "wav" in result:
                wav = result["wav"]
            elif "audio" in result:
                wav = result["audio"]
            else:
                wav = None
            if wav is None:
                raise RuntimeError("XTTS result dictionary did not contain audio samples")
            return self._to_numpy_audio(wav)

        if isinstance(result, tuple) and len(result) >= 1:
            return self._to_numpy_audio(result[0])

        if isinstance(result, list):
            if not result:
                raise RuntimeError("XTTS returned an empty audio list")
            return self._pick_audio_from_result(result[0])

        return self._to_numpy_audio(result)

    @staticmethod
    def _to_numpy_audio(wav: Any) -> np.ndarray:
        if isinstance(wav, torch.Tensor):
            wav_np = wav.detach().float().cpu().numpy()
        else:
            wav_np = np.asarray(wav, dtype=np.float32)

        if wav_np.ndim > 1:
            wav_np = np.squeeze(wav_np)
        if wav_np.ndim != 1:
            raise RuntimeError(f"Expected 1D audio array, got shape {wav_np.shape}")
        return wav_np.astype(np.float32, copy=False)

    def clone_voice(
        self,
        reference_wav_path: str,
        text: str,
        speed: float = 1.0,
        temperature: float = 0.7,
    ) -> Tuple[np.ndarray, int]:
        if self.model is None:
            raise RuntimeError("XTTS model is not loaded")

        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[reference_wav_path],
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

        kwargs = {
            "text": text,
            "language": "en",
            "gpt_cond_latent": gpt_cond_latent,
            "speaker_embedding": speaker_embedding,
            "length_penalty": self.model.config.length_penalty,
            "repetition_penalty": self.model.config.repetition_penalty,
            "top_k": self.model.config.top_k,
            "top_p": self.model.config.top_p,
            "temperature": temperature,
        }

        call_kwargs = self._filter_kwargs(self.model.inference, kwargs)
        result = self.model.inference(**call_kwargs)
        return self._pick_audio_from_result(result), self.output_sr

    def convert_voice(self, source_wav_path: str, reference_wav_path: str, text: Optional[str] = None) -> Tuple[np.ndarray, int]:
        # XTTS does not expose direct voice conversion; generate a second variant.
        if not text:
            raise RuntimeError("XTTS convert_voice fallback requires text")
        return self.clone_voice(
            reference_wav_path=reference_wav_path,
            text=text,
            speed=0.98,
            temperature=0.85,
        )

    def normalize_audio(self, audio_np: np.ndarray) -> np.ndarray:
        if audio_np.size == 0:
            raise RuntimeError("Cannot normalize empty audio")
        peak = float(np.max(np.abs(audio_np)))
        if peak <= 1e-8:
            return audio_np.astype(np.float32, copy=False)
        target_peak = 10.0 ** (-3.0 / 20.0)
        normalized = (audio_np / peak) * target_peak
        return normalized.astype(np.float32, copy=False)

    def save_wav(self, audio_np: np.ndarray, output_path: str | Path, sr: int) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio_np, sr, subtype="PCM_16")
