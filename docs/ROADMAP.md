# CortexMesh Research Roadmap

This document captures the current v1 direction for CortexMesh. The goal is to
make the project useful as a small research kit for concept compression, gated
routing, and differentiable external memory.

## Coordination Notes

This roadmap is intentionally forward-looking. Items below are research
directions unless they are also documented in the README, architecture notes,
evaluation notes, examples, or tests as current behavior.

When several agents are editing CortexMesh at once, keep roadmap changes narrow:
record planned work here, but avoid presenting planned features as shipped
capabilities. Current implementation notes belong in `README.md`,
`docs/ARCHITECTURE.md`, and `docs/EVALUATION.md`.

## Architecture Ideas

1. Holographic memory binding
   - Add a memory path that binds key/value concepts with vector-symbolic
     operations such as circular binding or learned bilinear binding.
   - Test with increasing key/value pairs and distractors.
   - Success metric: recall accuracy improves as prompts become longer.

2. Richer concept graph variants
   - The first fixed local `ConceptGraphMixer` is implemented.
   - Next variants can add rule-built edges, separator links, or same-symbol
     links while keeping the same strict no-attention constraint.
   - Success metric: better long-distance key/value and structured sequence
     tasks without transformer mechanics.

3. Rule capsules
   - Add small gated experts for symbolic transformations such as increment,
     alternation, cycle, copy, and recall.
   - Use rule labels already present in synthetic batches as auxiliary signal.
   - Success metric: better rule accuracy and better generalization to longer
     digit sequences.

4. Iterative concept relaxation
   - Treat routing cycles as a settling process and track cycle-to-cycle delta.
   - Add light stability metrics before adding any regularization.
   - Success metric: later cycles improve predictions while concept states stay
     numerically stable.

5. Discrete concept prototypes
   - Quantize some concept vectors into learned prototypes for interpretability.
   - Track prototype usage by task family.
   - Success metric: useful internal codes without dead prototypes or degraded
     recall.

## Training Roadmap

Start by measuring skills separately instead of relying only on total loss:

- token loss and token accuracy;
- rule accuracy;
- recall accuracy;
- reconstruction MSE;
- memory slot entropy;
- read-weight concentration;
- cycle delta norm;
- train tokens per second;
- batch-1 CPU latency.

Recommended v1 curriculum:

1. Sanity CPU tasks with short sequences and fixed seeds.
2. Digit rules: increment, skip, alternation, cycle, copy, reverse.
3. Key/value memory with 2, 4, then 8 pairs.
4. Composition: store a rule, recall a value, then apply the rule.
5. Controlled text from tiny grammars.
6. Out-of-distribution checks with longer lengths and extra distractors.

Initial loss direction:

```text
total =
  1.00 * next_token
+ 0.10 * reconstruction
+ 0.25 * rule
+ 0.35 * recall
+ 0.02 * memory_slot_balance
+ 0.03 * cycle_consistency
```

The extra slot and cycle terms should be added only after baseline metrics are
stable, so the project can tell whether they genuinely help.

## Product Roadmap

Near-term work:

- compare `CharNGramBaseline` against CortexMesh on the same text-corpus splits.
- add persisted benchmark result examples after the metrics stabilize.
- add richer curriculum phases for composed rule + memory tasks.
- add visual reports for memory read weights and cycle deltas.
- document longer-running experiments separately from smoke tests.

Medium-term work:

- benchmark against additional simple baselines such as MLP sequence models.
- ablations without memory, without routing cycles, and with varied slot counts.
- custom dataset adapter for user-provided character sequences.

Positioning:

```text
CortexMesh is a small PyTorch research architecture for studying concept
compression, gated routing, and differentiable external memory on CPU-friendly
sequence tasks.
```

It is not a general assistant model yet, and it should not be presented as a
frontier model competitor before strong evidence exists.
