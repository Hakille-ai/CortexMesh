from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cortexmesh import CharTokenizer, CortexMesh

from examples._common import build_small_trained_model, run_cli


def inspect_prompt(model: CortexMesh, tokenizer: CharTokenizer, prompt: str) -> dict[str, object]:
    """Return compact read-weight and memory statistics for one prompt."""

    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt or " ")
    tokens = torch.tensor([ids[-model.config.max_seq_len :]], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(tokens)

    read_weights = output["read_weights"][0].detach().cpu()
    memory = output["memory"][0].detach().cpu()
    row_sums = read_weights.sum(dim=-1)
    entropy = -(read_weights.clamp_min(1e-9) * read_weights.clamp_min(1e-9).log()).sum(dim=-1)
    top_weights, top_slots = read_weights.max(dim=-1)
    visible_prompt = tokenizer.decode(ids[-model.config.max_seq_len :])

    return {
        "prompt": visible_prompt,
        "token_count": len(visible_prompt),
        "read_weights_shape": tuple(read_weights.shape),
        "memory_shape": tuple(memory.shape),
        "read_weight_row_sum_min": float(row_sums.min()),
        "read_weight_row_sum_max": float(row_sums.max()),
        "read_entropy_mean": float(entropy.mean()),
        "memory_l2_mean": float(memory.norm(dim=-1).mean()),
        "memory_l2_max": float(memory.norm(dim=-1).max()),
        "memory_abs_mean": float(memory.abs().mean()),
        "top_reads": [
            {
                "token": token,
                "slot": int(slot),
                "weight": round(float(weight), 4),
            }
            for token, slot, weight in zip(visible_prompt, top_slots, top_weights)
        ],
    }


def run(steps: int = 6, seed: int = 13) -> dict[str, object]:
    model, tokenizer, _ = build_small_trained_model(steps=steps, seed=seed)
    return inspect_prompt(model, tokenizer, "a=3;b=7;c=1;?b=")


if __name__ == "__main__":
    run_cli("Train a tiny CortexMesh model and inspect memory reads.", run, default_steps=6)
