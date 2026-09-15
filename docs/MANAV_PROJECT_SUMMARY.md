# Manav Bidawat: VoiceLab project summary

**Period:** January-April 2026  
**Team:** four members, equal contribution (25% each)  
**GitHub:** [manav-bidawat](https://github.com/manav-bidawat)

## Resume summary

Contributed equally to a four-person audio deepfake detection and voice-cloning project using Python, PyTorch, and FastAPI. The project combines audio preprocessing, two detector checkpoints, a browser interface, and training and evaluation tools.

## Software engineering discussion points

| Area | Project evidence |
|---|---|
| API design | Upload, classification, generation, history, and health routes in `webapp/main.py`; typed request and response data models |
| Modular code | Separate data loaders, model definitions, training, evaluation, and benchmarking under `src/` |
| Performance measurement | Parameter counts, multiply-accumulate operation counts, and timed inference in `src/benchmark.py` |
| Error handling | Input validation, structured API errors, and model-readiness checks in `webapp/main.py` |
| Testing and packaging | Static repository tests, Python compilation in CI, Docker configuration, and optional XTTS dependencies |

These are features of the shared project. The repository does not establish individual ownership of these components.

## Benchmark example

The saved Run 2 results compare RawNet (1.14 million parameters, 2.22 ms mean inference time) with SpecRNet (0.80 million parameters, 1.61 ms). Both were measured on an RTX A5000 over 100 iterations. These are model timings rather than API response times.

Sources: [RawNet](../reports/run2/rawnet/benchmark.json), [SpecRNet](../reports/run2/specrnet/benchmark.json).
