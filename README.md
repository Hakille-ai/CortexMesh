# CortexMesh v0

[![Tests](https://github.com/Hakille-ai/CortexMesh/actions/workflows/tests.yml/badge.svg)](https://github.com/Hakille-ai/CortexMesh/actions/workflows/tests.yml)

CortexMesh is a small PyTorch research prototype for testing a non-transformer
sequence model. It processes character tokens through concept compression,
external memory slots, and gated routing cycles.

It is intentionally modest: the bundled tasks are synthetic, the default models
are CPU-friendly, and the project is meant to make architecture experiments easy
to inspect rather than to compete with production language models.

## What It Is

CortexMesh is built around five custom blocks:

1. `SignalEncoder` embeds token ids and adds simple time traits.
2. `ConceptField` compresses continuous signals into smaller concept vectors.
3. `MemoryLattice` writes concepts into differentiable memory slots and reads
   them back through learned gates.
4. `RouteMixer` updates concepts by mixing local concepts, memory readouts, and
   whole-sequence statistics.
5. `PredictionHead` produces next-token logits, rule-class logits, and
   memory-recall logits.

The model is strict about what it is not:

- no Q/K/V attention projections;
- no self-attention over token pairs;
- no multi-head attention;
- no transformer positional encoding stack;
- no recurrent hidden-state module such as RNN, LSTM, or GRU;
- no convolution layers.

PyTorch is used for tensors, modules, autograd, and optimization.

## Project Layout

```text
src/cortexmesh/
  config.py      model dimensions and validation
  data.py        tokenizer, synthetic tasks, and custom text corpus batches
  baselines.py   lightweight character n-gram baseline
  modules.py     SignalEncoder, ConceptField, MemoryLattice, RouteMixer, head
  model.py       CortexMesh assembly and forward pass
  trainer.py     CPU-friendly synthetic training loop
  evaluation.py  accuracy and internal-state metrics
  inspection.py  reusable memory/concept inspection helpers
  inference.py   generate_text, solve_rule_task, recall_memory helpers
  experimental.py standalone research components
  train_text.py  package CLI for custom text training
  demo.py        command-line demo

examples/
  text_demo.py    tiny text generation example
  rule_demo.py    digit-rule classification and next-symbol example
  memory_demo.py  key/value memory recall example
  custom_text_dataset.py train on an inline string or local text file
  inspect_memory.py read-weight and memory inspection example
  save_and_reload.py local persistence roundtrip example
  _common.py      shared example setup

benchmarks/
  run_benchmark.py reproducible CPU benchmark smoke runner

tests/
  test_*.py
```

## Install

From the repository root:

```powershell
python -m pip install -e .
```

For tests, install the test extra:

```powershell
python -m pip install -e ".[test]"
```

## Development Workflow

Recommended local loop from the repository root:

```powershell
python -m pip install -e ".[test]"
python -m pytest
python -m cortexmesh.demo --steps 2 --batch-size 4 --seed 13
python benchmarks/run_benchmark.py --config tiny --steps 2 --batch-size 4
```

Use the short demo and benchmark commands as CPU smoke checks before longer
experiments. They verify that training, evaluation, examples, and benchmark
plumbing still run without making claims about model quality.

## Quick Start

Run the full demo:

```powershell
python -m cortexmesh.demo --steps 30 --batch-size 16 --seed 11
```

After editable install, the console command is also available:

```powershell
cortexmesh-demo --steps 30 --batch-size 16 --seed 11
```

On Windows, if the Python user `Scripts` directory is not on `PATH`, use
`python -m cortexmesh.demo` or call the generated `cortexmesh-demo.exe` by its
full path.

The demo trains briefly on three synthetic task families:

- short text-like character sequences;
- digit sequences with simple hidden rules;
- key/value prompts such as `a=3;b=7;c=1;?b=`.

It then prints a training loss change plus one sample from each inference
helper:

```text
CortexMesh v0 demo
train loss: 4.1234 -> 3.5678
eval loss:  4.2345 -> 3.9876 (-0.2469)
text: cortex ...
rule: {'prompt': '02468', 'predicted_next': '...', 'predicted_rule_class': ...}
memory: {'prompt': 'a=3;b=7;c=1;?b=', 'recalled': '...', 'next_symbol': '...'}
```

Exact outputs vary with seed, number of steps, and PyTorch version. A short demo
is a smoke test, not proof of strong task performance.

## Examples

Each example builds a small model, trains for a few CPU steps, then calls one
inference helper.

```powershell
python examples/text_demo.py --steps 20 --seed 13
python examples/rule_demo.py --steps 20 --seed 13
python examples/memory_demo.py --steps 20 --seed 13
```

Use fewer steps for a fast smoke test:

```powershell
python examples/text_demo.py --steps 2
```

Train on your own small text corpus:

```powershell
python examples/custom_text_dataset.py --text "CortexMesh can train on local text." --steps 20
python examples/custom_text_dataset.py --text-file path/to/corpus.txt --steps 20 --save-dir checkpoints/custom-text
python examples/custom_text_dataset.py --text-file path/to/corpus.txt --steps 20 --batch-size 8 --seq-len 24 --prompt "cortex"
```

The package CLI exposes the same training path:

```powershell
python -m cortexmesh.train_text --text "CortexMesh can train on local text." --steps 20 --json-output
cortexmesh-train-text --text-file path/to/corpus.txt --steps 20 --save-dir checkpoints/custom-text
```

Inspect memory usage and test save/reload:

```powershell
python examples/inspect_memory.py --steps 2
python examples/save_and_reload.py --steps 2
```

Run the minimal CPU benchmark:

```powershell
python benchmarks/run_benchmark.py --config tiny --steps 2 --batch-size 4
python benchmarks/run_benchmark.py --config small --steps 5 --batch-size 8 --eval-batches 1 --json-output reports/bench-small.json
```

## Minimal API

```python
from cortexmesh import CortexMesh, CortexMeshConfig, CharTokenizer, Trainer

tokenizer = CharTokenizer()
config = CortexMeshConfig(vocab_size=tokenizer.vocab_size)
model = CortexMesh(config)

trainer = Trainer(model, tokenizer)
report = trainer.train_steps(steps=30, batch_size=16)

print(report["first_loss"], report["last_loss"])
```

Use a custom text corpus with the same trainer:

```python
from cortexmesh import TextCorpusFactory

text = "CortexMesh can learn from a tiny local corpus. " * 4
tokenizer = CharTokenizer.from_text(text)
config = CortexMeshConfig(vocab_size=tokenizer.vocab_size)
model = CortexMesh(config)
factory = TextCorpusFactory(text, tokenizer, seq_len=config.max_seq_len)
trainer = Trainer(model, tokenizer, factory=factory, eval_factory=factory)
report = trainer.train_steps(steps=20, batch_size=8)
```

Save and reload a local model:

```python
model.save_pretrained("checkpoints/cortexmesh-demo")
same_model = CortexMesh.from_pretrained("checkpoints/cortexmesh-demo")
```

Evaluate one synthetic batch:

```python
from cortexmesh import SyntheticTaskFactory, evaluate_batch

batch = SyntheticTaskFactory(tokenizer, seq_len=config.max_seq_len).make_batch(8)
metrics = evaluate_batch(model, batch)
print(metrics)
```

Use curriculum batches and a simple n-gram baseline:

```python
from cortexmesh import CharNGramBaseline, CurriculumTaskFactory

curriculum = CurriculumTaskFactory(tokenizer, seq_len=config.max_seq_len)
batch = curriculum.make_batch(8, step=250, fixed_cycle=True)

baseline = CharNGramBaseline(max_order=4).fit("cortex mesh memory " * 8)
print(baseline.generate("cortex ", steps=16))
```

Inspect model internals from code:

```python
from cortexmesh import summarize_prompt

report = summarize_prompt(model, tokenizer, "a=3;b=7;c=1;?b=")
print(report["memory_shape"], report["top_reads"][:3])
```

Inference helpers:

```python
from cortexmesh import generate_text, recall_memory, solve_rule_task

text = generate_text(model, tokenizer, "cortex ", steps=20)
rule = solve_rule_task(model, tokenizer, "02468")
memory = recall_memory(model, tokenizer, "a=3;b=7;?b=")
```

## Forward Pass Contract

`CortexMesh.forward(tokens)` expects integer token ids with shape
`[batch, time]`.

By default it returns:

- `logits`: `[batch, time, vocab_size]`
- `rule_logits`: `[batch, rule_classes]`
- `recall_logits`: `[batch, vocab_size]`
- `reconstruction`: `[batch, time, signal_dim]`
- `initial_reconstruction`: `[batch, time, signal_dim]`
- `summary`: `[batch, concept_dim]`
- `signals`: `[batch, time, signal_dim]`
- `concepts`: `[batch, time, concept_dim]`
- `memory`: `[batch, memory_slots, concept_dim]`
- `readout`: `[batch, time, concept_dim]`
- `read_weights`: `[batch, time, memory_slots]`
- `cycle_states`: `[cycles, batch, time, concept_dim]`

Pass `return_internal=False` when you only need prediction outputs and summary.

## Training Objective

`Trainer` uses synthetic batches from `SyntheticTaskFactory` and combines four
loss terms:

- next-token cross entropy;
- signal reconstruction MSE;
- rule-class cross entropy for rule examples;
- recall cross entropy for memory examples.

This keeps the demo self-contained. There is no external dataset download.

## Tests

```powershell
python -m pytest
```

The tests include coverage for:

- forward-pass output shapes;
- `MemoryLattice` read/write behavior;
- short fixed-batch CPU training;
- training reports with evaluation breakdowns;
- reproducible synthetic batches with metadata;
- curriculum synthetic batches;
- custom text-corpus batches;
- character n-gram baseline;
- persistence roundtrips;
- evaluation metrics;
- inspection helpers;
- experimental holographic memory;
- benchmark smoke execution;
- example execution.

The same command is used by the GitHub Actions workflow in
`.github/workflows/tests.yml` on Python 3.10, 3.11, and 3.12.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): current model path, internal tensors,
  and task surface.
- [Evaluation](docs/EVALUATION.md): local checks, measured losses, and current
  benchmark limits.
- [Roadmap](docs/ROADMAP.md): research ideas and future evaluation work.

## Research Roadmap

The current v1 research notes live in [docs/ROADMAP.md](docs/ROADMAP.md).
They prioritize memory binding, richer synthetic curricula, separate skill
metrics, CPU benchmarks, and a clearer research-kit API.

When multiple agents are working in the repository, treat the roadmap as a
coordination note rather than a claim that every listed item is implemented.
Docs should describe current behavior separately from planned experiments.

## License

CortexMesh is available under the MIT License. See [LICENSE](LICENSE).

## Current Limits

- The tokenizer is character-level and tiny.
- The datasets are synthetic and intentionally small.
- Short training runs are mostly smoke tests.
- The architecture is experimental and should be evaluated carefully before any
  serious use.
