from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cortexmesh import generate_text

from examples._common import build_small_trained_model, run_cli


def run(steps: int = 10, seed: int = 13) -> str:
    model, tokenizer, _ = build_small_trained_model(steps=steps, seed=seed)
    return generate_text(model, tokenizer, "mesh ", steps=16, temperature=0.7)


if __name__ == "__main__":
    run_cli("Train a tiny CortexMesh model and generate text.", run)
