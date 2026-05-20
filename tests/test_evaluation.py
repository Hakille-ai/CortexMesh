from __future__ import annotations

import math

import torch

from cortexmesh.config import CortexMeshConfig
from cortexmesh.data import CharTokenizer, SyntheticTaskFactory
from cortexmesh.evaluation import (
    compute_recall_accuracy,
    compute_rule_accuracy,
    compute_token_accuracy,
    cycle_delta_norm,
    evaluate_batch,
    memory_slot_entropy,
)
from cortexmesh.model import CortexMesh


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


def test_accuracy_helpers_use_logits_and_masks() -> None:
    batch = {
        "targets": torch.tensor([[1, 2, 3], [0, 1, 2]]),
        "rule_labels": torch.tensor([1, 3]),
        "rule_mask": torch.tensor([True, False]),
        "recall_targets": torch.tensor([4, 2]),
        "recall_mask": torch.tensor([False, True]),
    }
    output = {
        "logits": _logits_from_predictions(torch.tensor([[1, 0, 3], [0, 1, 1]]), classes=5),
        "rule_logits": _logits_from_predictions(torch.tensor([1, 0]), classes=4),
        "recall_logits": _logits_from_predictions(torch.tensor([0, 2]), classes=5),
    }

    assert math.isclose(compute_token_accuracy(output, batch), 4 / 6, rel_tol=1e-6)
    assert compute_rule_accuracy(output, batch) == 1.0
    assert compute_recall_accuracy(output, batch) == 1.0


def test_masked_accuracy_is_nan_when_mask_is_empty() -> None:
    batch = {
        "rule_labels": torch.tensor([1, 3]),
        "rule_mask": torch.tensor([False, False]),
        "recall_targets": torch.tensor([4, 2]),
        "recall_mask": torch.tensor([False, False]),
    }
    output = {
        "rule_logits": torch.zeros(2, 4),
        "recall_logits": torch.zeros(2, 5),
    }

    assert math.isnan(compute_rule_accuracy(output, batch))
    assert math.isnan(compute_recall_accuracy(output, batch))


def test_internal_metrics_are_finite_and_slot_entropy_uses_read_weights() -> None:
    output = {
        "read_weights": torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
            ]
        ),
        "cycle_states": torch.tensor(
            [
                [[[0.0, 0.0], [1.0, 1.0]]],
                [[[3.0, 4.0], [1.0, 2.0]]],
            ]
        ),
    }

    assert math.isclose(memory_slot_entropy(output), torch.log(torch.tensor(2.0)).item())
    assert math.isclose(cycle_delta_norm(output), 3.0)


def test_evaluate_batch_runs_on_cpu_and_restores_training_mode() -> None:
    torch.manual_seed(11)
    model, tokenizer = small_model()
    model.train()
    batch = SyntheticTaskFactory(tokenizer, seq_len=10, seed=11).make_batch(6, fixed_cycle=True)

    metrics = evaluate_batch(model, batch)

    assert model.training
    assert set(metrics) == {
        "token_accuracy",
        "rule_accuracy",
        "recall_accuracy",
        "memory_slot_entropy",
        "cycle_delta_norm",
    }
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    assert 0.0 <= metrics["rule_accuracy"] <= 1.0
    assert 0.0 <= metrics["recall_accuracy"] <= 1.0
    assert metrics["memory_slot_entropy"] >= 0.0
    assert metrics["cycle_delta_norm"] >= 0.0


def _logits_from_predictions(predictions: torch.Tensor, classes: int) -> torch.Tensor:
    logits = torch.zeros(*predictions.shape, classes)
    return logits.scatter_(-1, predictions.unsqueeze(-1), 1.0)
