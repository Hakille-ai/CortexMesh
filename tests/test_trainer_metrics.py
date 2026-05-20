from __future__ import annotations

import math

import torch

from cortexmesh.config import CortexMeshConfig
from cortexmesh.data import CharTokenizer
from cortexmesh.evaluation import (
    compute_recall_accuracy,
    compute_rule_accuracy,
    compute_token_accuracy,
    cycle_delta_norm,
    memory_slot_entropy,
)
from cortexmesh.model import CortexMesh
from cortexmesh.trainer import Trainer


def small_model() -> tuple[CortexMesh, CharTokenizer]:
    tokenizer = CharTokenizer()
    config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=16,
        concept_dim=12,
        memory_slots=4,
        cycles=2,
        route_hidden_dim=24,
        max_seq_len=10,
    )
    return CortexMesh(config), tokenizer


def test_evaluate_keeps_metrics_opt_in_and_restores_training_mode() -> None:
    torch.manual_seed(21)
    model, tokenizer = small_model()
    trainer = Trainer(model, tokenizer, seq_len=model.config.max_seq_len)
    model.train()

    report = trainer.evaluate(batches=1, batch_size=6)

    assert model.training
    assert "metrics" not in report
    assert report["batches"] == 1
    assert report["batch_size"] == 6


def test_evaluate_includes_mean_metrics_over_batches() -> None:
    torch.manual_seed(22)
    model, tokenizer = small_model()
    trainer = Trainer(model, tokenizer, seq_len=model.config.max_seq_len)
    rng_state = trainer.eval_factory.rng.getstate()
    expected = []

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            batch = trainer.eval_factory.make_batch(5, trainer.device, fixed_cycle=True).as_dict()
            output = model(batch["inputs"])
            expected.append(
                {
                    "token_accuracy": compute_token_accuracy(output, batch),
                    "rule_accuracy": compute_rule_accuracy(output, batch),
                    "recall_accuracy": compute_recall_accuracy(output, batch),
                    "memory_slot_entropy": memory_slot_entropy(output),
                    "cycle_delta_norm": cycle_delta_norm(output),
                }
            )
    trainer.eval_factory.rng.setstate(rng_state)

    report = trainer.evaluate(batches=2, batch_size=5, include_metrics=True)
    metrics = report["metrics"]

    assert set(metrics) == {
        "token_accuracy",
        "rule_accuracy",
        "recall_accuracy",
        "memory_slot_entropy",
        "cycle_delta_norm",
    }
    for key in metrics:
        expected_mean = sum(item[key] for item in expected) / len(expected)
        assert math.isclose(metrics[key], expected_mean, rel_tol=1e-6, abs_tol=1e-6)
