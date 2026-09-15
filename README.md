# VoiceLab: audio deepfake detection and voice cloning

VoiceLab is a four-person project completed from January to April 2026. It combines audio spoof detection, XTTS voice cloning, and experiments on ASVspoof 2019 Logical Access (LA) and Physical Access (PA) data.

## Team

All four members share equal project credit (25% each):

| Contributor | Credit |
|---|---:|
| [Sparshjoshi-iit](https://github.com/Sparshjoshi-iit) | 25% |
| [hasan72341](https://github.com/hasan72341) | 25% |
| [manav-bidawat](https://github.com/manav-bidawat) | 25% |
| [Ekansh2406](https://github.com/Ekansh2406) | 25% |

Git was set up after the project was completed. The initial import credits all four contributors; it does not record the original development timeline. See [team contributions](CONTRIBUTORS.md).

## What it does

- Accepts WAV/MP3 uploads and converts them to 16 kHz mono audio.
- Runs two AASIST-style detector checkpoints, trained on LA and PA respectively.
- Generates speech from reference audio and text when XTTS dependencies and weights are available.
- Shows audio playback, classification results, recent jobs, and server status in a browser.
- Includes training, evaluation, latency benchmarking, speaker-similarity analysis, and noisy-audio experiments.

Job and classification history are stored in memory and reset when the server restarts. XTTS weights and training datasets are not included.

## Run the detector

Use Python 3.11. From the project root on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r webapp/requirements.txt
VOICELAB_ENABLE_TTS=false python webapp/main.py --host 127.0.0.1 --port 7860
```

Open [localhost:7860](http://127.0.0.1:7860).

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`, then run:

```powershell
$env:VOICELAB_ENABLE_TTS = "false"
python webapp/main.py --host 127.0.0.1 --port 7860
```

The two detector checkpoints are in `models/checkpoints/`. Check whether they loaded:

```bash
curl http://127.0.0.1:7860/health
```

`classifier_loaded` reports detector readiness. `model_loaded` refers to XTTS and is `false` when generation is disabled.

## Docker

```bash
cp .env.example .env
docker compose up --build
```

See the [deployment guide](deployment/README.md) for environment variables and XTTS setup.

## Train and evaluate

Download ASVspoof 2019 LA and PA separately. The manifest builder expects the extracted folders under `LA/LA/` and `PA/PA/` in the data root.

```bash
python -m src.data.make_manifests --data-root . --out-dir data/manifests --verify-exists
python -m src.train \
  --data-root . \
  --train-manifest data/manifests/la_train.csv \
  --val-manifest data/manifests/la_dev.csv \
  --output-dir models/run1/rawnet \
  --model rawnet \
  --epochs 20 \
  --batch-size 64 \
  --trim-silence \
  --pre-emphasis \
  --augment \
  --balance-data
```

The training CLI supports `cnn`, `crnn`, `rawnet`, and `specrnet`. AASIST training is in `AASIST_FINAL.ipynb`. AudioMamba has no implementation in this repository.

The web runtime requirements do not cover every research script. Training also needs the datasets and dependencies imported by the selected script. Notebook paths need to be adjusted for the local dataset and checkpoint locations.

## Saved benchmarks

These are saved Run 2 measurements on an NVIDIA RTX A5000, averaged over 100 iterations. They measure model inference, excluding upload and API processing.

| Model | Parameters | MACs | Mean inference time | Checkpoint size |
|---|---:|---:|---:|---:|
| RawNet | 1,138,666 | 8.20 billion | 2.22 ms | 13.13 MB |
| SpecRNet | 804,771 | 536 million | 1.61 ms | 3.11 MB |

Sources: [RawNet benchmark](reports/run2/rawnet/benchmark.json), [SpecRNet benchmark](reports/run2/specrnet/benchmark.json). Detection metrics and ROC/DET plots are under `reports/`.

## Repository

| Path | Contents |
|---|---|
| `webapp/` | FastAPI server, detector wrapper, XTTS wrapper, and browser UI |
| `src/` | Data loaders, model definitions, training, evaluation, and research scripts |
| `models/checkpoints/` | Two AASIST detector checkpoints |
| `manifests/` | Dataset CSVs and experiment configuration |
| `reports/` | Saved evaluation metrics, benchmarks, and plots |
| `scripts/` | Training and evaluation shell scripts |
| `tests/` | Static repository checks |
| `docker/`, `deployment/` | Container definition and setup instructions |

## Checks

```bash
python -m compileall -q src webapp tests
python -m unittest discover -s tests -v
```

These checks cover syntax and repository files. They do not run model inference or reproduce the experiments.

## Notes

- [Architecture](docs/PROJECT_AUDIT.md)
- [Known limitations](docs/GAP_ANALYSIS.md)
- [Models and hardware](docs/MODEL_AND_GPU_REPORT.md)
- [Deployment design](docs/DEPLOYMENT_STRATEGY.md)
- [Remaining work](docs/IMPROVEMENTS_ROI.md)
