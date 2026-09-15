from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List

DEFAULT_TEXTS: List[str] = [
    "The quick brown fox jumps.",
    "Artificial intelligence is transforming modern industries.",
    "The signal was corrupted by unexpected noise, which significantly affected accuracy.",
    "She sells seashells.",
    "Please repeat the sentence clearly and confidently.",
    "Speech signals contain rich information about linguistic content and speaker identity.",
    "This is a test.",
    "The experiment yielded interesting and useful results.",
    "The system must generalize well even when evaluated on unseen datasets.",
    "He answered quickly.",
    "Timing and rhythm are important in natural speech.",
    "The pitch contour varies significantly across speakers and emotional states.",
    "The system works well.",
    "The dataset contains recordings from multiple speakers.",
    "Energy fluctuations across frames provide cues for distinguishing speech types.",
    "The signal is clean.",
    "The algorithm needs further improvement and tuning.",
    "The model struggles when exposed to real-world noisy environments.",
    "She spoke softly.",
    "The quick brown fox jumps over the lazy dog.",
    "The signal processing pipeline must be robust enough to handle distortions.",
    "The model predicts correctly.",
    "The weather today is calm and slightly pleasant.",
    "The classifier predicts whether a sample is real or artificially generated.",
    "This is another sample.",
    "The pitch contour varies across different speakers.",
    "The system needs further optimization for different acoustic conditions.",
    "Please speak clearly.",
    "The audio clip is five seconds long.",
    "Results must be reproducible across multiple runs.",
    "He paused briefly before answering the question.",
    "The speech rate varies across speakers depending on context.",
    "The articulation is clear.",
    "The system requires optimization.",
    "Articulation clarity can significantly affect how speech is perceived.",
    "He whispered something unclear.",
    "The system is currently under evaluation for reliability.",
    "Thank you for participating.",
    "The experiment was successful.",
    "The feature extraction process plays a crucial role in performance.",
    "End of evaluation.",
    "The data must be handled carefully.",
    "The model performance is stable.",
    "The prediction is uncertain.",
    "The feature extraction is complete.",
    "The classifier output is binary.",
    "The system is under evaluation.",
    "The signal is clean and clear.",
    "This concludes the sequence for testing the antispoofing system.",
    "Evaluation complete.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch voice cloning for DL1 dataset")
    parser.add_argument("--dl1-dir", type=Path, default=Path("DL1"), help="Directory with 1.wav..50.wav")
    parser.add_argument("--output-dir", type=Path, default=Path("ClonedVoice"), help="Output directory for cloned wav files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--backend",
        type=str,
        default="coqui",
        choices=["coqui", "command"],
        help="Cloning backend",
    )
    parser.add_argument(
        "--tts-command-template",
        type=str,
        default="",
        help=(
            "Required for backend=command. Placeholders: "
            "{text}, {reference_audio}, {output_wav}, {index}, {speaker_id}"
        ),
    )
    parser.add_argument(
        "--coqui-model",
        type=str,
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui model name for backend=coqui",
    )
    parser.add_argument(
        "--speaker1-ref",
        type=Path,
        default=None,
        help="Optional reference wav for files 1..25 (default: DL1/1.wav)",
    )
    parser.add_argument(
        "--speaker2-ref",
        type=Path,
        default=None,
        help="Optional reference wav for files 26..50 (default: DL1/26.wav)",
    )
    parser.add_argument(
        "--reference-mode",
        type=str,
        default="grouped",
        choices=["grouped", "per_file", "single"],
        help=(
            "Reference strategy: grouped (1..25 -> spk1, 26..50 -> spk2), "
            "per_file (index i uses DL1/i.wav), single (all use --single-reference)."
        ),
    )
    parser.add_argument(
        "--single-reference",
        type=Path,
        default=None,
        help="Reference wav used when --reference-mode single (default: DL1/1.wav)",
    )
    return parser.parse_args()


def _speaker_id(index: int) -> int:
    return 1 if index <= 25 else 2


def _get_reference(
    index: int,
    dl1_dir: Path,
    speaker1_ref: Path | None,
    speaker2_ref: Path | None,
    reference_mode: str,
    single_reference: Path | None,
) -> Path:
    if reference_mode == "per_file":
        return dl1_dir / f"{index}.wav"
    if reference_mode == "single":
        return single_reference if single_reference is not None else (dl1_dir / "1.wav")
    if index <= 25:
        return speaker1_ref if speaker1_ref is not None else (dl1_dir / "1.wav")
    return speaker2_ref if speaker2_ref is not None else (dl1_dir / "26.wav")


def _validate_inputs(dl1_dir: Path) -> None:
    missing = [str(i) for i in range(1, 51) if not (dl1_dir / f"{i}.wav").exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files in DL1: {', '.join(missing)}")
    if len(DEFAULT_TEXTS) != 50:
        raise ValueError(f"Expected 50 texts, found {len(DEFAULT_TEXTS)}")


def _run_command_backend(index: int, text: str, reference_audio: Path, output_wav: Path, template: str) -> None:
    if not template.strip():
        raise ValueError("--tts-command-template is required when --backend command")
    cmd = template.format(
        text=text,
        reference_audio=str(reference_audio.resolve()),
        output_wav=str(output_wav.resolve()),
        index=index,
        speaker_id=_speaker_id(index),
    )
    proc = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Command backend failed for {index}.wav: {err}")
    if not output_wav.exists():
        raise RuntimeError(f"Command backend did not produce file: {output_wav}")


def _run_coqui_backend(
    dl1_dir: Path,
    output_dir: Path,
    device: str,
    model_name: str,
    speaker1_ref: Path | None,
    speaker2_ref: Path | None,
    reference_mode: str,
    single_reference: Path | None,
) -> None:
    try:
        from TTS.api import TTS  # type: ignore
    except Exception as exc:
        raise RuntimeError("Coqui TTS is not installed. Install it with: pip install TTS") from exc

    tts = TTS(model_name=model_name).to(device)

    for index, text in enumerate(DEFAULT_TEXTS, start=1):
        reference_audio = _get_reference(
            index,
            dl1_dir,
            speaker1_ref,
            speaker2_ref,
            reference_mode,
            single_reference,
        )
        output_wav = output_dir / f"{index}.wav"
        tts.tts_to_file(
            text=text,
            speaker_wav=str(reference_audio),
            language="en",
            file_path=str(output_wav),
        )


def main() -> None:
    args = parse_args()

    dl1_dir = args.dl1_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_inputs(dl1_dir)

    speaker1_ref = args.speaker1_ref.resolve() if args.speaker1_ref else None
    speaker2_ref = args.speaker2_ref.resolve() if args.speaker2_ref else None
    single_reference = args.single_reference.resolve() if args.single_reference else None

    failures: List[Dict[str, str]] = []

    if args.backend == "coqui":
        try:
            _run_coqui_backend(
                dl1_dir=dl1_dir,
                output_dir=output_dir,
                device=args.device,
                model_name=args.coqui_model,
                speaker1_ref=speaker1_ref,
                speaker2_ref=speaker2_ref,
                reference_mode=args.reference_mode,
                single_reference=single_reference,
            )
        except Exception as exc:
            raise RuntimeError(f"Coqui backend failed: {exc}") from exc
    else:
        for index, text in enumerate(DEFAULT_TEXTS, start=1):
            output_wav = output_dir / f"{index}.wav"
            reference_audio = _get_reference(
                index,
                dl1_dir,
                speaker1_ref,
                speaker2_ref,
                args.reference_mode,
                single_reference,
            )
            try:
                _run_command_backend(index, text, reference_audio, output_wav, args.tts_command_template)
            except Exception as exc:
                failures.append({"index": str(index), "error": str(exc)})

    generated = sorted(output_dir.glob("*.wav"), key=lambda x: int(x.stem))
    summary = {
        "dl1_dir": str(dl1_dir),
        "output_dir": str(output_dir),
        "backend": args.backend,
        "device": args.device,
        "reference_mode": args.reference_mode,
        "generated_files": len(generated),
        "expected_files": 50,
        "failures": failures,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc())
        raise
