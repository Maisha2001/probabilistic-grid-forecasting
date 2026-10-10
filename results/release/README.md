# Release verification

The CPU-only Python 3.11 environment passed all 50 tests in 16.49 seconds and
the dependency check. Four browser tests passed against both the synthetic CI
fixture and the real trained models. The strict TypeScript build passed;
desktop, forecast and mobile views were visually inspected.

`service-validation.json` checks 384 node forecasts from the published GPU-fitted
snapshots through the CPU API. The largest difference is 0.002571 MW. It also
checks every observation scenario, coherent endpoints and custom histories.
Warm request medians are 25.94 ms for trees and 20.22 ms for the transformer;
each request predicts four horizons for six nodes. These are sequential ASGI
measurements on four CPU threads, including serialization but excluding network
latency, with 20 requests per family. They are not concurrent-load guarantees.

`reproduction.json`, `cpu-metrics.json` and `cpu-seal.json` record a complete
CPU-only reproduction: 290,304 forecast rows in 69.93 seconds, then 20,160 stress
rows in 16.85 seconds. Runtime excludes package installation, data preparation
and tracking. Resume preserved the version, predictions and tracking run.
The same CPU computation using the CUDA-distributed PyTorch build produced
identical predictions. GPU training produced different transformer epochs and
results; its earlier published evaluation remains unchanged.

`cpu-service-validation.json` verifies the reproduced models through the API.
Its 384 compared forecasts match saved CPU predictions to numerical precision.
The CPU-only transformer scores MASE 1.200 and 89.9% joint coverage, with mean
width 4,428 MW; the tree matches the published forward scores. No settings were
chosen from these reproduction results.

Windows tests used a short writable temporary path because this checkout's
deeply nested paths can exceed MLflow artifact-path limits. The model guide
documents that constraint. These files record local checks; GitHub Actions
independently runs the suites and browser integration on Linux.
