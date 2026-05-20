"""Evaluation helpers for CortexMesh model outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


BatchLike = Mapping[str, torch.Tensor] | Any


def compute_token_accuracy(output: Mapping[str, torch.Tensor], batch: BatchLike) -> float:
    """Compute next-token accuracy from model logits and batch targets."""

    data = _batch_dict(batch)
    predictions = output["logits"].argmax(dim=-1)
    return _mean_accuracy(predictions, data["targets"])


def compute_rule_accuracy(output: Mapping[str, torch.Tensor], batch: BatchLike) -> float:
    """Compute rule classification accuracy on examples selected by rule_mask."""

    data = _batch_dict(batch)
    return _masked_class_accuracy(
        output["rule_logits"],
        data["rule_labels"],
        data["rule_mask"],
    )


def compute_recall_accuracy(output: Mapping[str, torch.Tensor], batch: BatchLike) -> float:
    """Compute memory recall accuracy on examples selected by recall_mask."""

    data = _batch_dict(batch)
    return _masked_class_accuracy(
        output["recall_logits"],
        data["recall_targets"],
        data["recall_mask"],
    )


def memory_slot_entropy(output: Mapping[str, torch.Tensor], eps: float = 1e-12) -> float:
    """Estimate entropy of memory slot usage from read weights or memory norms."""

    if "read_weights" in output:
        read_weights = output["read_weights"].detach().float()
        slot_scores = read_weights.mean(dim=tuple(range(read_weights.ndim - 1)))
    else:
        memory = output["memory"].detach().float()
        slot_scores = memory.norm(dim=-1).mean(dim=0)

    probabilities = slot_scores / slot_scores.sum().clamp_min(eps)
    entropy = -(probabilities * (probabilities + eps).log()).sum()
    return float(entropy.cpu())


def cycle_delta_norm(output: Mapping[str, torch.Tensor]) -> float:
    """Compute the mean L2 change between consecutive recurrent cycle states."""

    states = output["cycle_states"].detach().float()
    if states.shape[0] < 2:
        return 0.0
    deltas = states[1:] - states[:-1]
    return float(deltas.norm(dim=-1).mean().cpu())


def evaluate_batch(model: torch.nn.Module, batch: BatchLike) -> dict[str, float]:
    """Run one no-grad evaluation pass and return accuracy/internal metrics."""

    data = _batch_dict(batch)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(data["inputs"])
            metrics = {
                "token_accuracy": compute_token_accuracy(output, data),
                "rule_accuracy": compute_rule_accuracy(output, data),
                "recall_accuracy": compute_recall_accuracy(output, data),
                "memory_slot_entropy": memory_slot_entropy(output),
                "cycle_delta_norm": cycle_delta_norm(output),
            }
    finally:
        if was_training:
            model.train()
    return metrics


def _batch_dict(batch: BatchLike) -> Mapping[str, torch.Tensor]:
    if isinstance(batch, Mapping):
        return batch
    if hasattr(batch, "as_dict"):
        return batch.as_dict()
    raise TypeError("batch must be a mapping or provide as_dict()")


def _mean_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    return float((predictions == targets).float().mean().cpu())


def _masked_class_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not bool(mask.any()):
        return float("nan")
    predictions = logits.argmax(dim=-1)
    return _mean_accuracy(predictions[mask], targets[mask])
