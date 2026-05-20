# CortexMesh Architecture

CortexMesh is a compact PyTorch sequence-modeling prototype. It is designed to
make concept compression, gated routing, and differentiable external memory easy
to inspect on small synthetic tasks.

The current model is not a transformer. It does not use Q/K/V attention
projections, multi-head attention, convolution layers, or recurrent modules such
as RNN, LSTM, or GRU.

## Forward Path

1. `SignalEncoder` maps character token ids to continuous signal vectors and
   adds simple time traits.
2. `ConceptField` compresses signals into lower-dimensional concept vectors and
   provides a reconstruction path for an auxiliary loss.
3. `MemoryLattice` writes concepts into learned memory slots and reads slot
   summaries back through learned gates.
4. `RouteMixer` runs for the configured number of routing cycles, mixing local
   concepts, memory readouts, and sequence-level statistics.
5. `PredictionHead` emits next-token logits, rule-class logits, recall logits,
   and a sequence summary.

`CortexMesh.forward(tokens)` expects integer token ids shaped `[batch, time]`.
When `return_internal=True`, it also returns intermediate tensors such as
signals, concepts, memory slots, read weights, and per-cycle states for
inspection.

## Current Task Surface

The bundled `SyntheticTaskFactory` generates three CPU-friendly task families:

- short character text sequences;
- digit-rule sequences with rule labels;
- key/value recall prompts with recall labels.

These tasks are intentionally small. They are useful for smoke tests,
architecture debugging, and early comparisons between internal variants, not for
claiming broad language-model capability.
