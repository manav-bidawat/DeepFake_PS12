# Run VoiceLab in Docker

From the project root:

```bash
cp .env.example .env
docker compose up --build
```

Open [localhost:7860](http://127.0.0.1:7860). The default configuration enables detection and disables XTTS generation.

## Detector configuration

```dotenv
VOICELAB_ENABLE_TTS=false
AASIST_CHECKPOINT_PATH=models/checkpoints/ModelA_LA_bestnew.pt
AASIST_CHECKPOINT_B_PATH=models/checkpoints/ModelB_PA_bestnew.pt
AASIST_THRESHOLD=0.420
AASIST_THRESHOLD_B=0.118
```

The repository includes both checkpoints. Classification is available only when both load.

## Check startup

```bash
curl http://127.0.0.1:7860/health
```

Check `classifier_loaded` for detector readiness. `model_loaded=false` is expected when XTTS is disabled. An HTTP response alone does not establish that the detector loaded successfully.

## Enable XTTS

For a local Python installation:

```bash
pip install -r webapp/requirements-tts.txt
export VOICELAB_ENABLE_TTS=true
export XTTS_MODEL_PATH=/absolute/path/to/xtts-model
python webapp/main.py --host 127.0.0.1 --port 7860
```

The model directory must contain `best_model.pth` or `model.pth`, `config.json`, and `vocab.json`.

The supplied Dockerfile installs only the detection runtime. To use XTTS in Docker, add the optional dependencies to the image and mount the model directory at the configured path. GPU use also requires a CUDA-compatible runtime and container GPU access.

## Storage

Uploads and generated audio live on local disk. Job and classification metadata live in memory and reset on restart. Check the Compose volume configuration before relying on file persistence.

See [known limitations](../docs/GAP_ANALYSIS.md) and [deployment design](../docs/DEPLOYMENT_STRATEGY.md).
