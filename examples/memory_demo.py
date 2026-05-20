from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cortexmesh import recall_memory

from examples._common import build_small_trained_model, run_cli


def run(steps: int = 10, seed: int = 13) -> dict[str, str]:
    model, tokenizer, _ = build_small_trained_model(steps=steps, seed=seed)
    return recall_memory(model, tokenizer, "a=3;b=7;c=1;?b=")


if __name__ == "__main__":
    run_cli("Train a tiny CortexMesh model and query key/value memory.", run)
