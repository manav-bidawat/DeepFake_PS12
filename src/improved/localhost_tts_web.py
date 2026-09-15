from __future__ import annotations

import argparse
import html
import random
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from scipy.io import wavfile

from src.improved.tts_postprocess_pipeline import (
    DacCodec,
    ProcessingContext,
    stage2_rir_convolution,
    stage3_dac_wash,
    stage4_band_limited_noise,
    stage5_resample_16k_sox,
    stage6_mp3_passthrough,
)


def build_app(
    rir_dir: Path,
    tts_command_template: str,
    work_dir: Path,
    device: str,
    seed: int,
) -> FastAPI:
    app = FastAPI(title="Local TTS + Post-Processing", version="1.0")

    rir_paths = list(rir_dir.rglob("*.wav")) + list(rir_dir.rglob("*.flac"))
    if not rir_paths:
        raise RuntimeError(f"No RIR files found in: {rir_dir}")

    if not tts_command_template.strip():
        raise RuntimeError(
            "Missing --tts-command-template. "
            "Provide your CosyVoice/VoiceCraft inference command with placeholders {text}, {reference_audio}, {output_wav}."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    dac_codec = DacCodec(device=device, bitrate_kbps=8)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return """
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Local TTS Pipeline</title>
    <style>
      body { font-family: sans-serif; max-width: 760px; margin: 24px auto; padding: 0 12px; }
      textarea, input, select, button { width: 100%; margin-top: 8px; margin-bottom: 16px; padding: 10px; }
      .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; }
    </style>
  </head>
  <body>
    <h2>Local Voice Clone + Post-Processing</h2>
    <div class=\"card\">
      <form action=\"/generate\" method=\"post\" enctype=\"multipart/form-data\">
        <label>Reference Voice Sample (.wav/.flac)</label>
        <input type=\"file\" name=\"reference_audio\" accept=\"audio/*\" required />

        <label>Target Text</label>
        <textarea name=\"text\" rows=\"5\" placeholder=\"Enter the text to synthesize\" required></textarea>

        <label>Output Format</label>
        <select name=\"output_format\">
          <option value=\"wav\" selected>WAV</option>
          <option value=\"mp3\">MP3</option>
        </select>

        <button type=\"submit\">Generate</button>
      </form>
    </div>
  </body>
</html>
        """

    @app.post("/generate")
    async def generate(
        text: str = Form(...),
        output_format: str = Form("wav"),
        reference_audio: UploadFile = File(...),
    ):
        if output_format not in {"wav", "mp3"}:
            raise HTTPException(status_code=400, detail="output_format must be wav or mp3")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is empty")

        job_id = uuid.uuid4().hex
        job_dir = work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        ref_suffix = Path(reference_audio.filename or "reference.wav").suffix or ".wav"
        reference_path = job_dir / f"reference{ref_suffix}"
        with reference_path.open("wb") as f:
            while True:
                chunk = await reference_audio.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

        stage1_path = job_dir / "stage1_tts.wav"
        stage2_path = job_dir / "stage2_rir.wav"
        stage3_path = job_dir / "stage3_dac.wav"
        stage4_path = job_dir / "stage4_noise.wav"
        stage5_path = job_dir / "stage5_16k.wav"
        stage6_path = job_dir / "stage6_mp3_passthrough.wav"

        cmd = tts_command_template.format(
            text=text,
            reference_audio=str(reference_path.resolve()),
            output_wav=str(stage1_path.resolve()),
            utt_id=job_id,
            input_audio=str(reference_path.resolve()),
        )
        proc = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        if proc.returncode != 0 or not stage1_path.exists():
            err = (proc.stderr or proc.stdout or "TTS failed").strip()
            raise HTTPException(
                status_code=500,
                detail=f"TTS failed. Command output: {html.escape(err)}",
            )

        ctx = ProcessingContext(
            data_root=work_dir,
            run_root=job_dir,
            rir_paths=rir_paths,
            noise_min_dbfs=-82.0,
            noise_max_dbfs=-78.0,
            target_sr=16000,
            device=device,
            rng=random.Random(seed),
        )

        try:
            stage2_rir_convolution(ctx, stage1_path, stage2_path)
            stage3_dac_wash(dac_codec, stage2_path, stage3_path)
            stage4_band_limited_noise(ctx, stage3_path, stage4_path)
            stage5_resample_16k_sox(stage4_path, stage5_path, 16000)
            stage6_mp3_passthrough(stage5_path, stage6_path, 128)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Post-processing failed: {exc}") from exc

        if output_format == "wav":
            return FileResponse(
                path=stage6_path,
                media_type="audio/wav",
                filename=f"{job_id}.wav",
            )

        mp3_path = job_dir / "final.mp3"
        cmd_mp3 = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(stage6_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(mp3_path),
        ]
        p = subprocess.run(cmd_mp3, check=False, capture_output=True)
        if p.returncode != 0 or not mp3_path.exists():
            raise HTTPException(status_code=500, detail="MP3 encoding failed")

        return FileResponse(path=mp3_path, media_type="audio/mpeg", filename=f"{job_id}.mp3")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "rir_count": len(rir_paths),
            "device": device,
            "work_dir": str(work_dir),
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local TTS + anti-spoof post-processing web app")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--rir-dir", type=Path, required=True, help="Directory with RIR wav/flac files")
    parser.add_argument("--tts-command-template", type=str, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("processed/web_jobs"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        rir_dir=args.rir_dir,
        tts_command_template=args.tts_command_template,
        work_dir=args.work_dir,
        device=args.device,
        seed=args.seed,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
