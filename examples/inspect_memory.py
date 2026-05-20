from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cortexmesh import CharTokenizer, CortexMesh
from cortexmesh.inspection import summarize_prompt

from examples._common import build_small_trained_model, run_cli


def inspect_prompt(model: CortexMesh, tokenizer: CharTokenizer, prompt: str) -> dict[str, object]:
    """Return compact read-weight and memory statistics for one prompt."""

    return summarize_prompt(model, tokenizer, prompt)


def run(steps: int = 6, seed: int = 13) -> dict[str, object]:
    model, tokenizer, _ = build_small_trained_model(steps=steps, seed=seed)
    return inspect_prompt(model, tokenizer, "a=3;b=7;c=1;?b=")


if __name__ == "__main__":
    run_cli("Train a tiny CortexMesh model and inspect memory reads.", run, default_steps=6)
