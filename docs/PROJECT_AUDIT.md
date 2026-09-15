# Architecture

VoiceLab has a FastAPI application, a static browser UI, and separate training and evaluation scripts.

## Request handling

`webapp/main.py` exposes these routes:

| Route | Purpose |
|---|---|
| `GET /` | Serve the frontend |
| `POST /upload-voice` | Accept WAV/MP3, convert to 16 kHz mono, and retain metadata |
| `POST /upload-text` | Accept UTF-8 text files |
| `POST /generate` | Generate speech with the XTTS wrapper |
| `POST /classify` | Run both detector checkpoints |
| `GET /classify-history` | Return up to 50 recent classifications |
| `GET /audio/{file_id}` | Stream generated audio |
| `GET /download/{file_id}` | Download generated audio |
| `GET /jobs` | Return up to 50 recent generation jobs |
| `DELETE /cleanup` | Remove files older than 24 hours |
| `GET /health` | Report model readiness and GPU memory |

Upload metadata, text metadata, jobs, and classification history live in process memory. Audio files live under `webapp/uploads/`, `webapp/outputs/`, and `webapp/temp/`.

## Detection

`webapp/aasist_classifier.py` loads the LA and PA checkpoints. Preprocessing converts audio to mono, resamples to 16 kHz, applies pre-emphasis and energy-based voice activity detection, normalizes RMS, and pads or crops to 64,600 samples.

The detector uses a SincConv frontend, a residual convolutional encoder, graph attention, and a two-class output. Default spoof thresholds are `0.420` for Model A and `0.118` for Model B.

Both checkpoints must load for the classification endpoint to be available. Thresholds are configuration values, not guarantees of accuracy on new recordings.

## Generation

`webapp/tts_engine.py` wraps Coqui XTTS. It loads a checkpoint, configuration, and vocabulary; computes conditioning from reference audio; generates English speech; and writes PCM-16 WAV output.

Generation is optional. The web app can start with it disabled through `VOICELAB_ENABLE_TTS=false`.

## Experiments

- `src/train.py`, `src/evaluate.py`, and `src/benchmark.py`: train detectors, calculate metrics, and measure inference cost.
- `AASIST_FINAL.ipynb`: train and compare LA and PA AASIST models.
- `q2_part1_calc_similarity_score_real_vs_syn.py` and `q2_part3_similarity_score_plot.py`: speaker-similarity analysis and plots.
- `q2_part2.ipynb`: attack-success evaluation on real and synthetic speech.
- `question3.ipynb`: AWGN and babble-noise evaluation at 20, 10, and 5 dB SNR.
- `src/improved/`: CNN experiments, XTTS fine-tuning and post-processing, and explanation methods based on audio perturbations.

Saved results are experiment artifacts. They have not been reproduced as part of the repository documentation cleanup.
