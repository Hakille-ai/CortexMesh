# CortexMesh v0

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
  data.py        character tokenizer and synthetic task generator
  modules.py     SignalEncoder, ConceptField, MemoryLattice, RouteMixer, head
  model.py       CortexMesh assembly and forward pass
  trainer.py     CPU-friendly synthetic training loop
  inference.py   generate_text, solve_rule_task, recall_memory helpers
  demo.py        command-line demo

examples/
  text_demo.py    tiny text generation example
  rule_demo.py    digit-rule classification and next-symbol example
  memory_demo.py  key/value memory recall example
  _common.py      shared example setup

tests/
  test_cortexmesh.py
```

## Install

From the repository root:

```powershell
python -m pip install -e .
```

For tests, install `pytest` if it is not already available:

```powershell
python -m pip install pytest
```

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

The current tests cover:

- forward-pass output shapes;
- `MemoryLattice` read/write behavior;
- short fixed-batch CPU training;
- training reports with evaluation breakdowns;
- reproducible synthetic batches with metadata;
- example execution.

## Research Roadmap

The current v1 research notes live in [docs/ROADMAP.md](docs/ROADMAP.md).
They prioritize memory binding, richer synthetic curricula, separate skill
metrics, CPU benchmarks, and a clearer research-kit API.

## Current Limits

- The tokenizer is character-level and tiny.
- The datasets are synthetic and intentionally small.
- Short training runs are mostly smoke tests.
- The architecture is experimental and should be evaluated carefully before any
  serious use.
