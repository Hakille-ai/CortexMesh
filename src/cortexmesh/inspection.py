"""Reusable inspection helpers for CortexMesh internals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def summarize_output(output: Mapping[str, torch.Tensor], batch_index: int = 0) -> dict[str, object]:
    """Return compact memory and concept statistics for a model output."""

    summary: dict[str, object] = {}

    if "read_weights" in output:
        read_weights = _select_batch(output["read_weights"], batch_index).detach().float().cpu()
        row_sums = read_weights.sum(dim=-1)
        entropy = _entropy(read_weights, dim=-1)
        summary.update(
            {
                "read_weights_shape": tuple(read_weights.shape),
                "read_weight_row_sum_min": float(row_sums.min()),
                "read_weight_row_sum_max": float(row_sums.max()),
                "read_entropy_mean": float(entropy.mean()),
            }
        )

    if "memory" in output:
        memory = _select_batch(output["memory"], batch_index).detach().float().cpu()
        memory_l2 = memory.norm(dim=-1)
        summary.update(
            {
                "memory_shape": tuple(memory.shape),
                "memory_l2_mean": float(memory_l2.mean()),
                "memory_l2_max": float(memory_l2.max()),
                "memory_abs_mean": float(memory.abs().mean()),
            }
        )

    if "concepts" in output:
        concepts = _select_batch(output["concepts"], batch_index).detach().float().cpu()
        concept_l2 = concepts.norm(dim=-1)
        summary.update(
            {
                "concepts_shape": tuple(concepts.shape),
                "concept_l2_mean": float(concept_l2.mean()),
                "concept_l2_max": float(concept_l2.max()),
                "concept_abs_mean": float(concepts.abs().mean()),
            }
        )

    if "cycle_states" in output:
        cycle_states = output["cycle_states"].detach().float().cpu()
        if cycle_states.ndim >= 4 and cycle_states.shape[0] > 1:
            deltas = cycle_states[1:, batch_index] - cycle_states[:-1, batch_index]
            summary["cycle_delta_l2_mean"] = float(deltas.norm(dim=-1).mean())
        else:
            summary["cycle_delta_l2_mean"] = 0.0

    return summary


def summarize_prompt(model: Any, tokenizer: Any, prompt: str, batch_index: int = 0) -> dict[str, object]:
    """Run one prompt through a model and return reusable inspection stats."""

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt or " ")
    visible_ids = ids[-model.config.max_seq_len :]
    tokens = torch.tensor([visible_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(tokens, return_internal=True)

    if was_training:
        model.train()

    visible_prompt = tokenizer.decode(visible_ids)
    report = {
        "prompt": visible_prompt,
        "token_count": len(visible_prompt),
        **summarize_output(output, batch_index=batch_index),
    }
    report["top_reads"] = top_memory_reads(
        output,
        tokens=visible_ids,
        tokenizer=tokenizer,
        batch_index=batch_index,
    )
    return report


def top_memory_reads(
    output: Mapping[str, torch.Tensor],
    tokens: Sequence[int] | torch.Tensor | str | None = None,
    tokenizer: Any | None = None,
    batch_index: int = 0,
    top_k: int = 1,
    digits: int = 4,
) -> list[dict[str, object]]:
    """Return top memory slots read at each sequence position."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if "read_weights" not in output:
        raise KeyError("output must include read_weights")

    read_weights = _select_batch(output["read_weights"], batch_index).detach().float().cpu()
    top_weights, top_slots = read_weights.topk(k=min(top_k, read_weights.shape[-1]), dim=-1)
    labels = _token_labels(tokens, tokenizer, read_weights.shape[0])

    reads: list[dict[str, object]] = []
    for position, label in enumerate(labels):
        slots = [
            {
                "slot": int(slot),
                "weight": round(float(weight), digits),
            }
            for slot, weight in zip(top_slots[position], top_weights[position])
        ]
        first = slots[0]
        entry: dict[str, object] = {
            "position": position,
            "token": label,
            "slot": first["slot"],
            "weight": first["weight"],
        }
        if top_k > 1:
            entry["top_slots"] = slots
        reads.append(entry)
    return reads


def _select_batch(tensor: torch.Tensor, batch_index: int) -> torch.Tensor:
    if tensor.ndim >= 3:
        return tensor[batch_index]
    return tensor


def _entropy(weights: torch.Tensor, dim: int) -> torch.Tensor:
    probs = weights.clamp_min(1e-9)
    return -(probs * probs.log()).sum(dim=dim)


def _token_labels(tokens: Sequence[int] | torch.Tensor | str | None, tokenizer: Any | None, count: int) -> list[str]:
    if tokens is None:
        return [str(index) for index in range(count)]
    if isinstance(tokens, str):
        return list(tokens[:count])
    if isinstance(tokens, torch.Tensor):
        token_ids = tokens.detach().cpu().tolist()
    else:
        token_ids = list(tokens)
    if tokenizer is not None:
        return list(tokenizer.decode(token_ids))[:count]
    return [str(int(token_id)) for token_id in token_ids[:count]]
