from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import os
import re
import math
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aasist_classifier import AASISTClassifier

try:
    from tts_engine import TTSEngine
except Exception as exc:  # pragma: no cover - enables detection-only deployments
    TTSEngine = None  # type: ignore[assignment]
    TTS_IMPORT_ERROR = exc
else:
    TTS_IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
CLEANUP_LOG = BASE_DIR / "cleanup.log"

MAX_AUDIO_SIZE = 50 * 1024 * 1024
MAX_TEXT_SIZE = 1 * 1024 * 1024
AUDIO_EXTS = {".wav", ".mp3"}
SURROGATE_RULES_PATH = Path(os.getenv("SURROGATE_RULES_PATH", str(BASE_DIR.parent / "explainability" / "global_surrogate" / "decision_tree_rules.txt")))
DEFAULT_TOP_SURROGATE_FEATURES = [
    "spectral_centroid_mean",
    "delta2_mfcc_2_std",
    "mfcc_7_mean",
    "delta_mfcc_5_mean",
    "delta2_mfcc_10_mean",
]
DEFAULT_XTTS_MODEL_DIR = BASE_DIR.parent / "experiments" / "dl1_xtts_ft" / "speaker2" / "run" / "training" / "GPT_XTTS_FT-April-19-2026_03+30AM-0000000"
DEFAULT_CHECKPOINT_DIR = BASE_DIR.parent / "models" / "checkpoints"

LOGGER = logging.getLogger("voicelab")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

app = FastAPI(title="VoiceLab", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AppError(Exception):
    def __init__(self, status_code: int, error: str, detail: str, code: str) -> None:
        self.status_code = status_code
        self.error = error
        self.detail = detail
        self.code = code
        super().__init__(detail)


class GenerationConfig(BaseModel):
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)


class GenerateRequest(BaseModel):
    voice_id: str
    text: Optional[str] = None
    text_id: Optional[str] = None
    mode: str = Field(default="clone")
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)


class VoiceMeta(BaseModel):
    voice_id: str
    path: str
    original_filename: str
    speaker_name: str
    duration_seconds: float
    sample_rate: int
    created_at: str


class TextMeta(BaseModel):
    text_id: str
    path: str
    content: str
    word_count: int
    created_at: str


class OutputMeta(BaseModel):
    file_id: str
    path: str
    filename: str
    label: str
    speaker_name: str
    created_at: str


class ClassificationMeta(BaseModel):
    classification_id: str
    filename: str
    results: Dict[str, Any]
    created_at: str


def _json_error(status_code: int, error: str, detail: str, code: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail, "code": code})


def _ensure_directories() -> None:
    for d in (UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, STATIC_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if UPLOAD_DIR.resolve() == OUTPUT_DIR.resolve():
        raise RuntimeError("uploads and outputs must be separate directories")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_stem(name: str) -> str:
    stem = Path(name).stem or "speaker"
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return sanitized[:64] if sanitized else "speaker"


def _gpu_memory_stats() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used = float(torch.cuda.memory_allocated() / (1024 * 1024))
    total = float(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    return used, total


def _parse_range_header(range_header: str, file_size: int) -> Tuple[int, int]:
    if not range_header.startswith("bytes="):
        raise ValueError("Invalid range unit")

    start_s, end_s = range_header.replace("bytes=", "", 1).split("-", 1)
    if start_s == "":
        suffix = int(end_s)
        start = max(file_size - suffix, 0)
        end = file_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1

    if start > end or start < 0 or end >= file_size:
        raise ValueError("Invalid byte range")
    return start, end


def _cleanup_old_files(hours: int = 24) -> int:
    cutoff = time.time() - (hours * 3600)
    deleted = 0

    for directory in (UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR):
        for path in directory.glob("**/*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to delete %s: %s", path, exc)

    CLEANUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CLEANUP_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} deleted_files={deleted}\n")

    return deleted


async def _periodic_cleanup_task(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=6 * 3600)
        except asyncio.TimeoutError:
            deleted = _cleanup_old_files(hours=24)
            LOGGER.info("Periodic cleanup done, deleted files: %s", deleted)


async def _save_upload_with_limit(upload: UploadFile, output_path: Path, max_size: int) -> int:
    size = 0
    with output_path.open("wb") as out_f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_size:
                out_f.close()
                output_path.unlink(missing_ok=True)
                raise AppError(413, "File too large", f"Uploaded file exceeds {max_size} bytes", "FILE_TOO_LARGE")
            out_f.write(chunk)
    return size


def _convert_to_wav16k_mono(source_path: Path, output_path: Path) -> Tuple[float, int]:
    waveform, sample_rate = torchaudio.load(str(source_path))
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    torchaudio.save(str(output_path), waveform, sample_rate)
    duration = float(waveform.shape[-1] / sample_rate)
    return duration, int(sample_rate)


def _resolve_text(req: GenerateRequest) -> str:
    if req.text_id:
        text_meta = app.state.text_store.get(req.text_id)
        if not text_meta:
            raise AppError(404, "Text not found", "text_id does not exist", "TEXT_NOT_FOUND")
        return text_meta["content"].strip()

    text = (req.text or "").strip()
    if not text:
        raise AppError(400, "Missing text", "Provide text or text_id", "TEXT_REQUIRED")
    return text


def _load_surrogate_top_features(top_k: int = 5) -> List[str]:
    if not SURROGATE_RULES_PATH.exists():
        return DEFAULT_TOP_SURROGATE_FEATURES[:top_k]

    text = SURROGATE_RULES_PATH.read_text(encoding="utf-8", errors="ignore")
    names: List[str] = []
    for line in text.splitlines():
        if "---" not in line or "class:" in line:
            continue
        part = line.split("---", 1)[1].strip()
        if "<=" in part:
            feat = part.split("<=", 1)[0].strip()
        elif ">" in part:
            feat = part.split(">", 1)[0].strip()
        else:
            continue
        if feat and feat not in names:
            names.append(feat)
        if len(names) >= top_k:
            break

    if not names:
        return DEFAULT_TOP_SURROGATE_FEATURES[:top_k]
    return names[:top_k]


def _default_xtts_model_path() -> Optional[str]:
    if DEFAULT_XTTS_MODEL_DIR.exists():
        return str(DEFAULT_XTTS_MODEL_DIR)
    return None


def _env_flag(name: str, default: str = "auto") -> str:
    return os.getenv(name, default).strip().lower()


def _load_wave_16k(path: Path) -> torch.Tensor:
    wave, sr = torchaudio.load(str(path))
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    if wave.size(0) > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sr != 16000:
        wave = torchaudio.functional.resample(wave, sr, 16000)
    return wave.float().cpu()


def _surrogate_feature_values(path: Path) -> Dict[str, float]:
    wave = _load_wave_16k(path)
    n_fft = 512
    hop = 160
    mfcc_tf = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={"n_fft": n_fft, "hop_length": hop, "n_mels": 40, "center": True},
    )
    mfcc = mfcc_tf(wave).squeeze(0)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)

    centroid = torchaudio.functional.spectral_centroid(
        wave,
        sample_rate=16000,
        pad=0,
        window=torch.hann_window(n_fft),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
    ).squeeze(0)

    def _safe_mean(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return 0.0
        flat = x.reshape(-1)
        finite = torch.isfinite(flat)
        if not bool(torch.any(finite)):
            return 0.0
        v = float(flat[finite].mean().item())
        return v if math.isfinite(v) else 0.0

    def _safe_std(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return 0.0
        flat = x.reshape(-1)
        finite = torch.isfinite(flat)
        if not bool(torch.any(finite)):
            return 0.0
        vals = flat[finite]
        v = float(vals.std(unbiased=False).item()) if vals.numel() > 1 else 0.0
        return v if math.isfinite(v) else 0.0

    values = {
        "spectral_centroid_mean": _safe_mean(centroid),
        "mfcc_2_mean": _safe_mean(mfcc[1]),
        "mfcc_7_mean": _safe_mean(mfcc[6]),
        "delta_mfcc_5_mean": _safe_mean(delta[4]),
        "delta2_mfcc_2_std": _safe_std(delta2[1]),
        "delta2_mfcc_10_mean": _safe_mean(delta2[9]),
    }
    return values


def _compare_surrogate_features(reference_path: Path, generated_path: Path) -> Dict[str, Any]:
    top_features = _load_surrogate_top_features(top_k=5)
    ref_vals = _surrogate_feature_values(reference_path)
    gen_vals = _surrogate_feature_values(generated_path)

    rows: List[Dict[str, Any]] = []
    for feat in top_features:
        if feat not in ref_vals or feat not in gen_vals:
            continue
        a = float(ref_vals[feat])
        b = float(gen_vals[feat])
        delta_abs = abs(a - b)
        denom = max(1e-6, abs(a) + abs(b))
        similarity = float(max(0.0, 1.0 - min(1.0, delta_abs / denom)))
        rows.append(
            {
                "feature": feat,
                "reference": round(a, 6),
                "generated": round(b, 6),
                "delta_abs": round(delta_abs, 6),
                "similarity": round(similarity, 4),
                "verdict": "similar" if similarity >= 0.7 else "dissimilar",
            }
        )

    overall = float(sum(r["similarity"] for r in rows) / len(rows)) if rows else 0.0
    similar_n = int(sum(1 for r in rows if r["verdict"] == "similar"))
    return {
        "top_features": rows,
        "overall_similarity": round(overall, 4),
        "similar_count": similar_n,
        "total_features": len(rows),
    }


def _infer_sync(req: GenerateRequest, voice_meta: Dict[str, Any], text: str, job_id: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine: TTSEngine = app.state.tts_engine
    outputs: List[Dict[str, Any]] = []

    clone_audio, clone_sr = engine.clone_voice(
        reference_wav_path=voice_meta["path"],
        text=text,
        speed=req.generation_config.speed,
        temperature=req.generation_config.temperature,
    )
    clone_audio = engine.normalize_audio(clone_audio)
    clone_path = OUTPUT_DIR / f"{job_id}_clone.wav"
    engine.save_wav(clone_audio, clone_path, clone_sr)

    clone_info = sf.info(str(clone_path))
    clone_file_id = uuid.uuid4().hex
    clone_output = {
        "label": "Cloned Voice",
        "file_id": clone_file_id,
        "filename": clone_path.name,
        "duration_seconds": float(clone_info.duration),
        "download_url": f"/download/{clone_file_id}",
        "stream_url": f"/audio/{clone_file_id}",
        "file_size_bytes": int(clone_path.stat().st_size),
    }
    try:
        clone_output["feature_similarity"] = _compare_surrogate_features(Path(voice_meta["path"]), clone_path)
    except Exception as exc:
        LOGGER.warning("Surrogate similarity failed for clone output: %s", exc)
    outputs.append(clone_output)

    app.state.output_store[clone_file_id] = OutputMeta(
        file_id=clone_file_id,
        path=str(clone_path),
        filename=clone_path.name,
        label="cloned",
        speaker_name=voice_meta["speaker_name"],
        created_at=_utc_now_iso(),
    ).model_dump()

    if req.mode == "both" and voice_meta["duration_seconds"] > 10.0:
        converted_audio, converted_sr = engine.convert_voice(
            source_wav_path=voice_meta["path"],
            reference_wav_path=voice_meta["path"],
            text=text,
        )
        converted_audio = engine.normalize_audio(converted_audio)
        converted_path = OUTPUT_DIR / f"{job_id}_converted.wav"
        engine.save_wav(converted_audio, converted_path, converted_sr)

        conv_info = sf.info(str(converted_path))
        conv_file_id = uuid.uuid4().hex
        conv_output = {
            "label": "Converted Voice",
            "file_id": conv_file_id,
            "filename": converted_path.name,
            "duration_seconds": float(conv_info.duration),
            "download_url": f"/download/{conv_file_id}",
            "stream_url": f"/audio/{conv_file_id}",
            "file_size_bytes": int(converted_path.stat().st_size),
        }
        try:
            conv_output["feature_similarity"] = _compare_surrogate_features(Path(voice_meta["path"]), converted_path)
        except Exception as exc:
            LOGGER.warning("Surrogate similarity failed for converted output: %s", exc)
        outputs.append(conv_output)

        app.state.output_store[conv_file_id] = OutputMeta(
            file_id=conv_file_id,
            path=str(converted_path),
            filename=converted_path.name,
            label="converted",
            speaker_name=voice_meta["speaker_name"],
            created_at=_utc_now_iso(),
        ).model_dump()

    return {
        "job_id": job_id,
        "status": "complete",
        "outputs": outputs,
        "processing_time_seconds": round(time.perf_counter() - t0, 3),
        "error": None,
    }


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return _json_error(exc.status_code, exc.error, exc.detail, exc.code)


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and {"error", "detail", "code"}.issubset(detail.keys()):
        return JSONResponse(status_code=exc.status_code, content=detail)
    return _json_error(exc.status_code, "HTTP error", str(detail), "HTTP_ERROR")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _json_error(422, "Validation error", str(exc), "VALIDATION_ERROR")


@app.exception_handler(Exception)
async def generic_error_handler(_: Request, exc: Exception) -> JSONResponse:
    traceback.print_exc()
    return _json_error(500, "Internal server error", "Unexpected server error", "INTERNAL_ERROR")


@app.on_event("startup")
async def startup_event() -> None:
    _ensure_directories()

    app.state.start_time = time.time()
    app.state.voice_store = {}
    app.state.text_store = {}
    app.state.output_store = {}
    app.state.jobs = []
    app.state.classification_history = []

    deleted = _cleanup_old_files(hours=24)
    LOGGER.info("Startup cleanup complete, deleted files: %s", deleted)

    model_path = os.getenv("XTTS_MODEL_PATH") or os.getenv("COSYVOICE_MODEL_PATH") or _default_xtts_model_path()
    checkpoint_path = os.getenv("XTTS_CHECKPOINT_PATH") or None
    config_path = os.getenv("XTTS_CONFIG_PATH") or None
    vocab_path = os.getenv("XTTS_VOCAB_PATH") or None
    tts_mode = _env_flag("VOICELAB_ENABLE_TTS", "auto")
    should_load_tts = tts_mode in {"1", "true", "yes", "on"} or (tts_mode == "auto" and bool(model_path))
    load_t0 = time.perf_counter()
    app.state.model_loaded = False
    app.state.model_error = None
    app.state.tts_engine = None
    app.state.generation_enabled = False
    if should_load_tts:
        if TTSEngine is None:
            app.state.model_error = f"TTS import failed: {TTS_IMPORT_ERROR}"
            LOGGER.error(app.state.model_error)
        else:
            try:
                app.state.tts_engine = TTSEngine(
                    model_path=model_path,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    vocab_path=vocab_path,
                )
                load_seconds = time.perf_counter() - load_t0
                used_mb, total_mb = _gpu_memory_stats()
                app.state.model_loaded = True
                app.state.generation_enabled = True
                LOGGER.info("TTS model loaded in %.2f seconds", load_seconds)
                LOGGER.info("GPU memory after TTS load: %.2f MB / %.2f MB", used_mb, total_mb)
            except Exception as exc:
                load_seconds = time.perf_counter() - load_t0
                app.state.model_error = str(exc)
                LOGGER.exception("TTS model load failed after %.2f seconds", load_seconds)
    else:
        app.state.model_error = "TTS disabled; set VOICELAB_ENABLE_TTS=true and XTTS paths to enable generation"
        LOGGER.info(app.state.model_error)

    app.state.classifier_a_loaded = False
    app.state.classifier_b_loaded = False
    app.state.classifier_a_error = None
    app.state.classifier_b_error = None
    app.state.classifier_a = None
    app.state.classifier_b = None

    classify_ckpt_a = os.getenv("AASIST_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_DIR / "ModelA_LA_bestnew.pt"))
    classify_ckpt_b = os.getenv("AASIST_CHECKPOINT_B_PATH", str(DEFAULT_CHECKPOINT_DIR / "ModelB_PA_bestnew.pt"))
    classify_threshold_a = float(os.getenv("AASIST_THRESHOLD", "0.420"))
    classify_threshold_b = float(os.getenv("AASIST_THRESHOLD_B", "0.118"))
    try:
        app.state.classifier_a = AASISTClassifier(
            checkpoint_path=classify_ckpt_a,
            threshold=classify_threshold_a,
        )
        app.state.classifier_a_loaded = True
        LOGGER.info("AASIST Model A loaded from %s with threshold %.3f", classify_ckpt_a, classify_threshold_a)
    except Exception as exc:
        app.state.classifier_a_error = str(exc)
        LOGGER.exception("AASIST Model A failed to load")

    try:
        app.state.classifier_b = AASISTClassifier(
            checkpoint_path=classify_ckpt_b,
            threshold=classify_threshold_b,
        )
        app.state.classifier_b_loaded = True
        LOGGER.info("AASIST Model B loaded from %s with threshold %.3f", classify_ckpt_b, classify_threshold_b)
    except Exception as exc:
        app.state.classifier_b_error = str(exc)
        LOGGER.exception("AASIST Model B failed to load")

    host = os.getenv("WEBAPP_HOST", "0.0.0.0")
    port = os.getenv("WEBAPP_PORT", "7860")
    LOGGER.info("VoiceLab running at http://%s:%s", host, port)

    stop_event = asyncio.Event()
    app.state.cleanup_stop_event = stop_event
    app.state.cleanup_task = asyncio.create_task(_periodic_cleanup_task(stop_event))


@app.on_event("shutdown")
async def shutdown_event() -> None:
    stop_event: asyncio.Event = app.state.cleanup_stop_event
    task: asyncio.Task = app.state.cleanup_task
    stop_event.set()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        # Expected when cancelling background cleanup on shutdown.
        pass
    except Exception:
        pass


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/upload-voice")
async def upload_voice(voice_file: UploadFile = File(...)) -> Dict[str, Any]:
    ext = Path(voice_file.filename or "").suffix.lower()
    content_type = (voice_file.content_type or "").lower()
    if ext not in AUDIO_EXTS or not content_type.startswith("audio/"):
        raise AppError(400, "Invalid voice file", "Only WAV/MP3 audio files are allowed", "INVALID_AUDIO_FILE")

    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_upload{ext or '.wav'}"
    await _save_upload_with_limit(voice_file, temp_path, MAX_AUDIO_SIZE)

    voice_id = uuid.uuid4().hex
    speaker_name = _safe_stem(voice_file.filename or "speaker")
    output_filename = f"{voice_id}_{speaker_name}.wav"
    output_path = UPLOAD_DIR / output_filename

    try:
        duration, sample_rate = _convert_to_wav16k_mono(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)

    meta = VoiceMeta(
        voice_id=voice_id,
        path=str(output_path),
        original_filename=voice_file.filename or output_filename,
        speaker_name=speaker_name,
        duration_seconds=duration,
        sample_rate=sample_rate,
        created_at=_utc_now_iso(),
    )
    app.state.voice_store[voice_id] = meta.model_dump()

    return {
        "voice_id": voice_id,
        "filename": output_filename,
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "message": "Voice file uploaded successfully",
    }


@app.post("/classify")
async def classify_audio(voice_file: UploadFile = File(...)) -> Dict[str, Any]:
    if not bool(getattr(app.state, "classifier_a_loaded", False)) or not bool(getattr(app.state, "classifier_b_loaded", False)):
        raise AppError(503, "Classifier unavailable", "AASIST Model A/B are not both loaded", "CLASSIFIER_NOT_LOADED")

    ext = Path(voice_file.filename or "").suffix.lower()
    content_type = (voice_file.content_type or "").lower()
    if ext not in AUDIO_EXTS or not content_type.startswith("audio/"):
        raise AppError(400, "Invalid voice file", "Only WAV/MP3 audio files are allowed", "INVALID_AUDIO_FILE")

    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_classify{ext or '.wav'}"
    normalized_path = TEMP_DIR / f"{uuid.uuid4().hex}_classify_16k.wav"
    await _save_upload_with_limit(voice_file, temp_path, MAX_AUDIO_SIZE)
    t0 = time.perf_counter()
    try:
        _convert_to_wav16k_mono(temp_path, normalized_path)
        loop = asyncio.get_running_loop()
        def _predict_both() -> Dict[str, Any]:
            model_a = app.state.classifier_a.predict_file(str(normalized_path))
            model_b = app.state.classifier_b.predict_file(str(normalized_path))
            model_a["model"] = "ModelA_LA_bestnew.pt"
            model_b["model"] = "ModelB_PA_bestnew.pt"
            return {"model_a": model_a, "model_b": model_b}

        result = await loop.run_in_executor(None, _predict_both)
    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower() or ("cuda" in msg.lower() and "memory" in msg.lower()):
            raise AppError(503, "GPU memory full, try shorter audio", "GPU memory full, try shorter audio", "GPU_OOM") from exc
        traceback.print_exc()
        raise AppError(500, "Classification failed", "AASIST inference failed", "CLASSIFICATION_FAILED") from exc
    except Exception as exc:
        traceback.print_exc()
        raise AppError(500, "Classification failed", "AASIST inference failed", "CLASSIFICATION_FAILED") from exc
    finally:
        temp_path.unlink(missing_ok=True)
        normalized_path.unlink(missing_ok=True)

    classification_id = uuid.uuid4().hex
    created_at = _utc_now_iso()
    record = ClassificationMeta(
        classification_id=classification_id,
        filename=voice_file.filename or "uploaded.wav",
        results=result,
        created_at=created_at,
    ).model_dump()
    app.state.classification_history.append(record)
    app.state.classification_history = app.state.classification_history[-50:]

    return {
        **record,
        "processing_time_seconds": round(time.perf_counter() - t0, 3),
    }


@app.get("/classify-history")
async def classify_history() -> List[Dict[str, Any]]:
    return list(reversed(app.state.classification_history[-50:]))


@app.post("/upload-text")
async def upload_text(text_file: UploadFile = File(...)) -> Dict[str, Any]:
    ext = Path(text_file.filename or "").suffix.lower()
    if ext != ".txt":
        raise AppError(400, "Invalid text file", "Only .txt files are allowed", "INVALID_TEXT_FILE")

    temp_path = TEMP_DIR / f"{uuid.uuid4().hex}_text.txt"
    await _save_upload_with_limit(text_file, temp_path, MAX_TEXT_SIZE)

    try:
        content = temp_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        temp_path.unlink(missing_ok=True)
        raise AppError(400, "Invalid encoding", "Text file must be UTF-8", "INVALID_TEXT_ENCODING") from exc

    text_id = uuid.uuid4().hex
    save_name = f"{text_id}_text.txt"
    save_path = UPLOAD_DIR / save_name
    save_path.write_text(content, encoding="utf-8")
    temp_path.unlink(missing_ok=True)

    words = [w for w in content.strip().split() if w]
    meta = TextMeta(
        text_id=text_id,
        path=str(save_path),
        content=content,
        word_count=len(words),
        created_at=_utc_now_iso(),
    )
    app.state.text_store[text_id] = meta.model_dump()

    return {
        "text_id": text_id,
        "content": content,
        "word_count": len(words),
        "message": "Text file uploaded successfully",
    }


@app.post("/generate")
async def generate(req: GenerateRequest) -> Dict[str, Any]:
    if not bool(getattr(app.state, "model_loaded", False)):
        raise AppError(
            503,
            "Model unavailable",
            getattr(app.state, "model_error", None) or "XTTS model is not loaded on this server",
            "MODEL_NOT_LOADED",
        )

    if req.mode not in {"clone", "both"}:
        raise AppError(400, "Invalid mode", "mode must be clone or both", "INVALID_MODE")

    voice_meta = app.state.voice_store.get(req.voice_id)
    if not voice_meta:
        raise AppError(404, "Voice not found", "voice_id does not exist", "VOICE_NOT_FOUND")

    if req.mode == "both" and not req.text_id:
        raise AppError(400, "Missing text file", "Mode both requires text_id from uploaded text file", "TEXT_FILE_REQUIRED")

    text = _resolve_text(req)
    if not text:
        raise AppError(400, "Empty text", "Text content is empty", "EMPTY_TEXT")

    job_id = uuid.uuid4().hex
    created_at = _utc_now_iso()
    app.state.jobs.append(
        {
            "job_id": job_id,
            "status": "processing",
            "created_at": created_at,
            "voice_filename": voice_meta["original_filename"],
            "text_preview": text[:80],
            "output_count": 0,
            "outputs": [],
        }
    )
    app.state.jobs = app.state.jobs[-50:]

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, functools.partial(_infer_sync, req, voice_meta, text, job_id))
    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower() or "cuda" in msg.lower() and "memory" in msg.lower():
            raise AppError(503, "GPU memory full, try shorter text", "GPU memory full, try shorter text", "GPU_OOM") from exc
        traceback.print_exc()
        raise AppError(500, "Inference failed", "XTTS inference failed", "INFERENCE_FAILED") from exc
    except Exception as exc:
        traceback.print_exc()
        raise AppError(500, "Inference failed", "XTTS inference failed", "INFERENCE_FAILED") from exc

    for job in reversed(app.state.jobs):
        if job["job_id"] == job_id:
            job["status"] = result["status"]
            job["output_count"] = len(result["outputs"])
            job["outputs"] = result["outputs"]
            break

    return result


@app.get("/audio/{file_id}")
async def stream_audio(file_id: str, request: Request) -> Response:
    meta = app.state.output_store.get(file_id)
    if not meta:
        raise AppError(404, "File not found", "Invalid file_id", "AUDIO_NOT_FOUND")

    path = Path(meta["path"])
    if not path.exists():
        raise AppError(404, "File not found", "Audio file no longer exists", "AUDIO_NOT_FOUND")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(str(path), media_type="audio/wav", headers={"Accept-Ranges": "bytes"})

    try:
        start, end = _parse_range_header(range_header, file_size)
    except ValueError:
        return _json_error(416, "Invalid range", "Requested byte range is not satisfiable", "INVALID_RANGE")

    chunk_size = 1024 * 64

    def file_iterator() -> Any:
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                data = f.read(to_read)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": "audio/wav",
    }
    return StreamingResponse(file_iterator(), status_code=206, headers=headers, media_type="audio/wav")


@app.get("/download/{file_id}")
async def download_audio(file_id: str) -> FileResponse:
    meta = app.state.output_store.get(file_id)
    if not meta:
        raise AppError(404, "File not found", "Invalid file_id", "DOWNLOAD_NOT_FOUND")

    path = Path(meta["path"])
    if not path.exists():
        raise AppError(404, "File not found", "Audio file no longer exists", "DOWNLOAD_NOT_FOUND")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "cloned" if meta["label"] == "cloned" else "converted"
    filename = f"{meta['speaker_name']}_{ts}_{label}.wav"
    return FileResponse(str(path), media_type="audio/wav", filename=filename)


@app.get("/jobs")
async def jobs() -> List[Dict[str, Any]]:
    return list(reversed(app.state.jobs[-50:]))


@app.delete("/cleanup")
async def cleanup() -> Dict[str, Any]:
    deleted = _cleanup_old_files(hours=24)
    return {"deleted_files": deleted}


@app.get("/health")
async def health() -> Dict[str, Any]:
    used_mb, total_mb = _gpu_memory_stats()
    a_loaded = bool(getattr(app.state, "classifier_a_loaded", False))
    b_loaded = bool(getattr(app.state, "classifier_b_loaded", False))
    return {
        "status": "ok",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
        "generation_enabled": bool(getattr(app.state, "generation_enabled", False)),
        "model_error": getattr(app.state, "model_error", None),
        "classifier_loaded": a_loaded and b_loaded,
        "classifier_a_loaded": a_loaded,
        "classifier_b_loaded": b_loaded,
        "classifier_a_error": getattr(app.state, "classifier_a_error", None),
        "classifier_b_error": getattr(app.state, "classifier_b_error", None),
        "gpu_memory_used_mb": round(used_mb, 2),
        "gpu_memory_total_mb": round(total_mb, 2),
        "uptime_seconds": round(time.time() - app.state.start_time, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--xtts-checkpoint", default=None)
    parser.add_argument("--xtts-config", default=None)
    parser.add_argument("--xtts-vocab", default=None)
    parser.add_argument("--aasist-checkpoint", default=None)
    parser.add_argument("--aasist-checkpoint-b", default=None)
    parser.add_argument("--aasist-threshold", type=float, default=0.420)
    parser.add_argument("--aasist-threshold-b", type=float, default=0.118)
    args = parser.parse_args()

    if args.model_path:
        os.environ["XTTS_MODEL_PATH"] = args.model_path
    if args.xtts_checkpoint:
        os.environ["XTTS_CHECKPOINT_PATH"] = args.xtts_checkpoint
    if args.xtts_config:
        os.environ["XTTS_CONFIG_PATH"] = args.xtts_config
    if args.xtts_vocab:
        os.environ["XTTS_VOCAB_PATH"] = args.xtts_vocab
    if args.aasist_checkpoint:
        os.environ["AASIST_CHECKPOINT_PATH"] = args.aasist_checkpoint
    if args.aasist_checkpoint_b:
        os.environ["AASIST_CHECKPOINT_B_PATH"] = args.aasist_checkpoint_b
    os.environ["AASIST_THRESHOLD"] = str(args.aasist_threshold)
    os.environ["AASIST_THRESHOLD_B"] = str(args.aasist_threshold_b)
    os.environ["WEBAPP_HOST"] = args.host
    os.environ["WEBAPP_PORT"] = str(args.port)

    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)
