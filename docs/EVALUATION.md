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

For a fast development loop, keep the commands small:

```powershell
python -m pytest
python -m cortexmesh.demo --steps 2 --batch-size 4 --seed 13
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

For local text experiments, `TextCorpusFactory` can turn an inline string or a
UTF-8 file into deterministic next-character batches:

```python
from cortexmesh import CharTokenizer, TextCorpusFactory

text = "tiny local corpus " * 8
tokenizer = CharTokenizer.from_text(text)
factory = TextCorpusFactory(text, tokenizer, seq_len=24, seed=13)
batch = factory.make_batch(4, fixed_cycle=True)
```

The matching example script exposes the same path from the command line:

```powershell
python examples/custom_text_dataset.py --text "tiny local corpus" --steps 20 --batch-size 8 --seq-len 24
python examples/custom_text_dataset.py --text-file path/to/corpus.txt --steps 20 --prompt "cortex" --save-dir checkpoints/custom-text
python -m cortexmesh.train_text --text "tiny local corpus" --steps 20 --json-output
```

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

`Trainer.evaluate(..., include_metrics=True)` includes the same metric family in
the trainer report, aggregated over evaluation batches.

For a non-neural sanity baseline, `CharNGramBaseline` provides a deterministic
character n-gram model:

```python
from cortexmesh import CharNGramBaseline

baseline = CharNGramBaseline(max_order=4).fit("tiny local corpus " * 8)
print(baseline.score_next_token_accuracy("tiny local corpus " * 2, seq_len=8))
```

## Benchmark Smoke Runner

For reproducible CPU smoke measurements, use:

```powershell
python benchmarks/run_benchmark.py --config tiny --steps 2 --batch-size 4
python benchmarks/run_benchmark.py --config small --steps 5 --batch-size 8 --eval-batches 1 --json-output reports/bench-small.json
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

This document only covers the local pytest, demo, metric helpers, n-gram
baseline, and benchmark smoke checks that are part of the current public
workflow. Saved benchmark artifacts, broader baseline comparisons, and long-run
quality claims should be documented alongside their integration. Related
evaluation ideas are tracked in `docs/ROADMAP.md`.
