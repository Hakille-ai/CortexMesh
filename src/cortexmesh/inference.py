"""Inference helpers for CortexMesh."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import CharTokenizer
from .model import CortexMesh


def _model_device(model: CortexMesh) -> torch.device:
    return next(model.parameters()).device


def generate_text(
    model: CortexMesh,
    tokenizer: CharTokenizer,
    prompt: str,
    steps: int = 40,
    temperature: float = 0.8,
) -> str:
    """Generate short text from a prompt."""

    model.eval()
    ids = tokenizer.encode(prompt or " ")
    device = _model_device(model)
    with torch.no_grad():
        for _ in range(steps):
            context = ids[-model.config.max_seq_len :]
            tokens = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(tokens, return_internal=False)["logits"][0, -1]
            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
    return tokenizer.decode(ids)


def solve_rule_task(
    model: CortexMesh,
    tokenizer: CharTokenizer,
    sequence: str,
) -> dict[str, object]:
    """Predict the next symbol and rule class for a digit-rule prompt."""

    model.eval()
    device = _model_device(model)
    ids = tokenizer.encode(sequence or "0")
    with torch.no_grad():
        tokens = torch.tensor([ids[-model.config.max_seq_len :]], dtype=torch.long, device=device)
        output = model(tokens, return_internal=False)
        next_id = int(torch.argmax(output["logits"][0, -1]).item())
        rule_id = int(torch.argmax(output["rule_logits"][0]).item())
    return {
        "prompt": sequence,
        "predicted_next": tokenizer.decode([next_id]),
        "predicted_rule_class": rule_id,
    }


def recall_memory(
    model: CortexMesh,
    tokenizer: CharTokenizer,
    memory_prompt: str,
) -> dict[str, str]:
    """Predict a recalled value from a key/value prompt."""

    model.eval()
    device = _model_device(model)
    ids = tokenizer.encode(memory_prompt or "a=0;?a=")
    with torch.no_grad():
        tokens = torch.tensor([ids[-model.config.max_seq_len :]], dtype=torch.long, device=device)
        output = model(tokens, return_internal=False)
        recall_id = int(torch.argmax(output["recall_logits"][0]).item())
        next_id = int(torch.argmax(output["logits"][0, -1]).item())
    return {
        "prompt": memory_prompt,
        "recalled": tokenizer.decode([recall_id]),
        "next_symbol": tokenizer.decode([next_id]),
    }
