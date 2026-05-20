from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cortexmesh import CharTokenizer, CortexMesh, recall_memory

from examples._common import build_small_trained_model, run_cli


def _logit_signature(model: CortexMesh, tokenizer: CharTokenizer, prompt: str) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt or " ")
    tokens = torch.tensor([ids[-model.config.max_seq_len :]], dtype=torch.long, device=device)
    with torch.no_grad():
        return model(tokens, return_internal=False)["logits"].detach().cpu()


def _find_save_reload_api(model: CortexMesh) -> Callable[[Path], CortexMesh] | None:
    model_type = type(model)

    if hasattr(model, "save_pretrained") and hasattr(model_type, "from_pretrained"):

        def save_pretrained_roundtrip(path: Path) -> CortexMesh:
            model.save_pretrained(path)
            return model_type.from_pretrained(path)

        return save_pretrained_roundtrip

    if hasattr(model, "save") and hasattr(model_type, "load"):

        def save_load_roundtrip(path: Path) -> CortexMesh:
            checkpoint = path / "cortexmesh"
            model.save(checkpoint)
            return model_type.load(checkpoint)

        return save_load_roundtrip

    return None


def run(steps: int = 4, seed: int = 13) -> dict[str, object]:
    model, tokenizer, report = build_small_trained_model(steps=steps, seed=seed)
    prompt = "a=3;b=7;c=1;?b="
    before = recall_memory(model, tokenizer, prompt)
    roundtrip = _find_save_reload_api(model)

    if roundtrip is None:
        return {
            "status": "unavailable",
            "message": (
                "No CortexMesh save/reload API was detected yet "
                "(looked for save_pretrained/from_pretrained and save/load)."
            ),
            "trained_steps": steps,
            "last_loss": report["last_loss"],
            "before": before,
        }

    before_logits = _logit_signature(model, tokenizer, prompt)
    with tempfile.TemporaryDirectory() as tmpdir:
        reloaded = roundtrip(Path(tmpdir))
        after = recall_memory(reloaded, tokenizer, prompt)
        after_logits = _logit_signature(reloaded, tokenizer, prompt)

    return {
        "status": "ok",
        "trained_steps": steps,
        "last_loss": report["last_loss"],
        "before": before,
        "after": after,
        "max_logit_delta": float((before_logits - after_logits).abs().max()),
    }


if __name__ == "__main__":
    run_cli("Train a tiny CortexMesh model, then save and reload if the API exists.", run, default_steps=4)
