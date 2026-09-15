# Models and hardware

## Included detector weights

| Checkpoint | Size | Default threshold |
|---|---:|---:|
| `models/checkpoints/ModelA_LA_bestnew.pt` | 460,986 bytes | 0.420 |
| `models/checkpoints/ModelB_PA_bestnew.pt` | 460,986 bytes | 0.118 |

The web detector can select CPU execution. Checkpoint size alone does not measure runtime memory: Python, PyTorch, audio decoding, activations, and simultaneous requests also consume memory.

## XTTS weights

XTTS weights are not included. Generation needs a directory containing `best_model.pth` or `model.pth`, plus `config.json` and `vocab.json`.

The wrapper selects CUDA when available and otherwise uses CPU. Measure load time, peak memory, and generation time with the actual checkpoint before choosing deployment hardware. This repository does not contain a CPU or GPU capacity test for the hosted generation service.

## Saved GPU benchmarks

Run 2 benchmark files record an NVIDIA RTX A5000 and 100 iterations:

| Model | Parameters | MACs | Mean latency | Checkpoint size |
|---|---:|---:|---:|---:|
| RawNet | 1,138,666 | 8,200,239,040 | 2.2203 ms | 13.1283 MB |
| SpecRNet | 804,771 | 536,367,504 | 1.6072 ms | 3.1051 MB |

Sources: [RawNet](../reports/run2/rawnet/benchmark.json) and [SpecRNet](../reports/run2/specrnet/benchmark.json).

These measurements exclude request handling and audio upload. They do not measure the web application's AASIST checkpoints. CPU performance and concurrent request capacity still need measurement.
