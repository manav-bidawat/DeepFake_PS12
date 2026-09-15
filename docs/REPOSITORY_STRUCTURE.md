# Repository structure

| Path | Purpose |
|---|---|
| `.github/workflows/ci.yml` | Python compilation and static tests |
| `configs/voicelab.example.yaml` | Example runtime and training settings |
| `deployment/README.md` | Container and model setup |
| `docker/Dockerfile` | CPU web application image |
| `docs/` | Architecture, limitations, model notes, and remaining work |
| `manifests/` | ASVspoof CSVs and run settings |
| `models/checkpoints/` | LA and PA AASIST checkpoints |
| `reports/` | Selected metrics, benchmarks, and ROC/DET plots |
| `scripts/` | Training and evaluation entry points |
| `src/data/` | Dataset loading, manifests, and augmentation |
| `src/models/` | CNN, RawNet, and SpecRNet model definitions |
| `src/improved/` | Additional training, TTS, and explainability experiments |
| `src/train.py` | Detector training CLI |
| `src/evaluate.py` | Detector evaluation CLI |
| `src/benchmark.py` | Model size and inference benchmarking |
| `tests/` | Static repository checks |
| `webapp/` | FastAPI app, inference wrappers, and static frontend |
| `CONTRIBUTORS.md` | Team credit and Git-history notes |

Root notebooks contain AASIST training, attack-success evaluation, and noisy-audio experiments. Root `q2_*.py` scripts calculate and plot speaker similarity.

Runtime uploads, outputs, and temporary files are ignored by Git. Full datasets, tensor caches, XTTS weights, experiment directories, and local tooling caches are also excluded.
