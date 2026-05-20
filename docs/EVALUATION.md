# CortexMesh Evaluation

The current evaluation path is intentionally lightweight and local. The test
suite checks that the architecture runs, reports losses, and executes the bundled
examples on CPU.

## Local Checks

Install the package with its test extra:

```powershell
python -m pip install -e ".[test]"
```

Run the test suite:

```powershell
python -m pytest
```

Run a short demo smoke test:

```powershell
python -m cortexmesh.demo --steps 30 --batch-size 16 --seed 11
```

## What Is Measured Today

`Trainer.evaluate(...)` returns a deterministic loss report over synthetic
batches. The report includes total loss and the current component losses:

- next-token cross entropy;
- reconstruction MSE;
- rule-class cross entropy on rule examples;
- recall cross entropy on memory examples.

The tests also verify reproducible synthetic batches with metadata. This makes
it possible to compare short experiments with fixed seeds, while keeping the
project free of external dataset downloads.

The `cortexmesh.evaluation` module also exposes lightweight metrics for a model
output or batch:

- `compute_token_accuracy`;
- `compute_rule_accuracy`;
- `compute_recall_accuracy`;
- `memory_slot_entropy`;
- `cycle_delta_norm`;
- `evaluate_batch`.

These helpers are intentionally small and operate on the tensors already
returned by `CortexMesh.forward(...)`.

## Benchmark Smoke Runner

For reproducible CPU smoke measurements, use:

```powershell
python benchmarks/run_benchmark.py --config tiny --steps 2 --batch-size 4
```

Useful options:

- `--config tiny|small`
- `--steps`
- `--batch-size`
- `--seed`
- `--eval-batches`
- `--json-output path/to/result.json`

The benchmark reports wall-clock training seconds, train losses, optional eval
losses, and the concrete model dimensions used for the run. It is meant for
local comparison between small CortexMesh variants, not for broad quality
claims.

## Not Covered Here

This document only covers the local pytest, demo, metric helpers, and benchmark
smoke checks that are part of the current public workflow. Saved benchmark
artifacts, baseline model comparisons, and long-run quality claims should be
documented alongside their integration. Related evaluation ideas are tracked in
`docs/ROADMAP.md`.
