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
5. Optional `ConceptGraphMixer` layers propagate concept messages over a fixed
   local graph of left/right neighbors and sequence anchors.
6. `PredictionHead` emits next-token logits, rule-class logits, recall logits,
   and a sequence summary.

`CortexMesh.forward(tokens)` expects integer token ids shaped `[batch, time]`.
When `return_internal=True`, it also returns intermediate tensors such as
signals, concepts, memory slots, read weights, and per-cycle states for
inspection.

## Optional Concept Graph Mixer

`CortexMeshConfig(graph_mix_layers=N, graph_radius=R)` enables local concept
graph propagation inside each routing cycle. The mixer is strict about the
non-transformer constraint:

- no Q/K/V projections;
- no pairwise learned attention matrix;
- no softmax over token pairs;
- no convolution or recurrent cell.

Instead, each position receives fixed left/right neighbor summaries within
`graph_radius`, plus sequence anchors from the first concept, last concept, and
mean concept. A learned gate decides how much of that local graph message should
update each concept.

When enabled and `return_internal=True`, the forward pass includes
`graph_states` shaped `[cycles * graph_mix_layers, batch, time, concept_dim]`.
The option defaults to `graph_mix_layers=0` so older configs and checkpoints
remain compatible.

## Current Task Surface

The bundled `SyntheticTaskFactory` generates three CPU-friendly task families:

- short character text sequences;
- digit-rule sequences with rule labels;
- key/value recall prompts with recall labels.

These tasks are intentionally small. They are useful for smoke tests,
architecture debugging, and early comparisons between internal variants, not for
claiming broad language-model capability.

`TextCorpusFactory` adds a small user-data path for local experiments. It slices
an inline string or text file into next-character batches compatible with
`Trainer`, using the same `[inputs, targets]` shape as the synthetic tasks while
leaving rule and recall masks disabled.

`CurriculumTaskFactory` wraps `SyntheticTaskFactory` for staged experiments.
Phase 0 emits short text-only batches, phase 1 mixes text and rules, and phase 2
mixes text, rules, and memory examples. It can be driven directly with
`make_batch(..., step=...)` or passed to `Trainer`, which forwards training step
information when a factory supports it.

Inspection helpers in `cortexmesh.inspection` summarize the internal tensors
returned by the forward pass. They keep visualization and debugging code out of
the core model while making memory slots, read weights, concept norms, and cycle
deltas easier to inspect.

## Current vs Planned Work

This document describes the implemented v0 path. Forward-looking items such as
new memory binding schemes, richer rule capsules, and broader benchmark suites
are tracked in `docs/ROADMAP.md` until they land in code, examples, and tests.
