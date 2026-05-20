from __future__ import annotations

import torch

from cortexmesh import CortexMesh, CortexMeshConfig, CharTokenizer, SyntheticTaskFactory, Trainer
from cortexmesh.modules import ConceptGraphMixer, MemoryLattice
from examples.memory_demo import run as run_memory_demo
from examples.rule_demo import run as run_rule_demo
from examples.text_demo import run as run_text_demo


def small_model() -> tuple[CortexMesh, CharTokenizer]:
    tokenizer = CharTokenizer()
    config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=24,
        concept_dim=16,
        memory_slots=6,
        cycles=2,
        route_hidden_dim=32,
        max_seq_len=16,
    )
    return CortexMesh(config), tokenizer


def test_forward_shapes() -> None:
    model, tokenizer = small_model()
    tokens = torch.randint(0, tokenizer.vocab_size, (3, 12))
    output = model(tokens)
    assert output["logits"].shape == (3, 12, tokenizer.vocab_size)
    assert output["reconstruction"].shape == (3, 12, model.config.signal_dim)
    assert output["memory"].shape == (3, model.config.memory_slots, model.config.concept_dim)
    assert output["rule_logits"].shape == (3, model.config.rule_classes)
    assert output["recall_logits"].shape == (3, tokenizer.vocab_size)
    assert "graph_states" not in output


def test_forward_with_concept_graph_mixer_shapes() -> None:
    tokenizer = CharTokenizer()
    config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=24,
        concept_dim=16,
        memory_slots=6,
        cycles=2,
        route_hidden_dim=32,
        max_seq_len=16,
        graph_mix_layers=2,
        graph_radius=1,
    )
    model = CortexMesh(config)
    tokens = torch.randint(0, tokenizer.vocab_size, (2, 10))

    output = model(tokens)

    assert output["logits"].shape == (2, 10, tokenizer.vocab_size)
    assert output["concepts"].shape == (2, 10, config.concept_dim)
    assert output["graph_states"].shape == (
        config.cycles * config.graph_mix_layers,
        2,
        10,
        config.concept_dim,
    )


def test_concept_graph_mixer_is_local_without_wraparound() -> None:
    mixer = ConceptGraphMixer(concept_dim=4, route_hidden_dim=8, radius=1)
    concepts = torch.arange(1 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4)

    left = mixer._directional_context(concepts, direction="left")
    right = mixer._directional_context(concepts, direction="right")

    assert torch.allclose(left[:, 0], torch.zeros_like(left[:, 0]))
    assert torch.allclose(left[:, 1], concepts[:, 0])
    assert torch.allclose(left[:, 3], concepts[:, 2])
    assert torch.allclose(right[:, 0], concepts[:, 1])
    assert torch.allclose(right[:, 2], concepts[:, 3])
    assert torch.allclose(right[:, 3], torch.zeros_like(right[:, 3]))


def test_memory_lattice_writes_and_reads() -> None:
    torch.manual_seed(0)
    lattice = MemoryLattice(concept_dim=8, memory_slots=4)
    concepts = torch.zeros(2, 5, 8)
    concepts[:, 0, 0] = 3.0
    concepts[:, 2, 3] = -2.0
    readout, memory, weights = lattice(concepts)
    assert readout.shape == concepts.shape
    assert memory.shape == (2, 4, 8)
    assert weights.shape == (2, 5, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-5)
    assert memory.abs().sum() > 0
    assert readout.abs().sum() > 0


def test_training_reduces_loss_on_fixed_batch() -> None:
    torch.manual_seed(3)
    model, tokenizer = small_model()
    trainer = Trainer(model, tokenizer, lr=5e-3, seq_len=model.config.max_seq_len)
    report = trainer.train_steps(steps=20, batch_size=9, fixed_batch=True)
    assert report["last_loss"] < report["first_loss"]
    assert report["eval_before"]["loss"] > report["eval_after"]["loss"]
    assert report["skipped_steps"] == 0


def test_training_report_includes_eval_breakdowns() -> None:
    torch.manual_seed(4)
    model, tokenizer = small_model()
    trainer = Trainer(model, tokenizer, lr=5e-3, seq_len=model.config.max_seq_len)
    report = trainer.train_steps(steps=2, batch_size=6, eval_batches=1)
    before = report["eval_before"]
    after = report["eval_after"]
    assert before["batches"] == 1
    assert after["batch_size"] == 6
    assert set(after["breakdown_dict"]) == {"total", "token", "reconstruction", "rule", "recall"}
    assert len(report["smoothed_losses"]) == 2


def test_synthetic_batches_are_seed_reproducible_with_metadata() -> None:
    tokenizer = CharTokenizer()
    first = SyntheticTaskFactory(tokenizer, seq_len=12, seed=19).make_batch(9, fixed_cycle=True)
    second = SyntheticTaskFactory(tokenizer, seq_len=12, seed=19).make_batch(9, fixed_cycle=True)

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)
    assert first.metadata == second.metadata
    assert first.metadata["task_counts"] == {"memory": 3, "rule": 3, "text": 3}
    assert len(set(first.metadata["variants"])) > 1

    plain = first.as_dict()
    assert "metadata" not in plain
    assert first.as_dict(include_metadata=True)["metadata"] == first.metadata


def test_examples_run_on_cpu() -> None:
    torch.manual_seed(5)
    text = run_text_demo(steps=2)
    rule = run_rule_demo(steps=2)
    memory = run_memory_demo(steps=2)
    assert isinstance(text, str) and len(text) > 0
    assert "predicted_next" in rule
    assert "recalled" in memory
