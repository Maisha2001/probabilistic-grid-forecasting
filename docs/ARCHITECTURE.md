# Architecture

```mermaid
flowchart LR
  A[Pinned CSV archive] --> B[DuckDB checks and hourly Parquet]
  B --> C[Chronological fitting and calibration]
  C --> D[Tree and transformer snapshots]
  D --> E[Forward evaluation and stress audit]
  E --> F[Checksum inventory and local MLflow]
  F --> G[Verified FastAPI service]
  G --> H[TypeScript forecast explorer]
```

`grid-pipeline` coordinates downloading, data validation, model fitting,
evaluation, version sealing and local tracking. Protected output roots and a
single-process lock prevent accidental overwrites. The archive is fixed and
there is no recurring ingestion requirement, so this application uses a local
restartable command rather than an always-running Airflow deployment. Docker
and a live scheduler are not requirements or validated deployment targets.

The data layer validates checksums, period completeness, finite readings and
node alignment. DuckDB performs hourly aggregation; Parquet preserves the
regular, typed series. Forecast features and calibration use observations
available before their respective boundaries. Evaluation labels remain separate.

Trees model each horizon's quantiles. A compact patch transformer predicts all
horizons jointly. Reconciliation combines regional and aggregate estimates;
calibration estimates a simultaneous band across leaves. Aggregated endpoints
are envelopes. Their measured coverage limitations are visible in the explorer.

Experiment inventories identify the dataset, protocol, outputs and fitted
artifacts. Service startup checks all of them and the inference-source hashes
recorded during fitting. Integrity checks detect changes; they do not establish
trust in an unknown model file or provide a cryptographic signature.

The service loads the two model snapshots and all three transformer seeds once
on CPU. A lock limits inference to one request at a time, with four compute
threads. This keeps the local demo's resource use bounded and avoids a GPU
dependency. Read-only metadata and results are served separately. Requests do
not write files, fit models or execute user code.

The TypeScript interface uses native SVG charts, accessible form labels, tables,
CSV export and responsive layouts. It has no runtime dependency on a third-party
API or CDN. Vite builds static assets mounted behind the same loopback service.
Playwright checks model selection, inference, input errors, exports, failed
requests and mobile layout. Backend tests check leakage boundaries, corruption,
causal missing-data repair, output coherence and request validation.
