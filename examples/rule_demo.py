from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cortexmesh import solve_rule_task

from examples._common import build_small_trained_model, run_cli


def run(steps: int = 10, seed: int = 13) -> dict[str, object]:
    model, tokenizer, _ = build_small_trained_model(steps=steps, seed=seed)
    return solve_rule_task(model, tokenizer, "02468")


if __name__ == "__main__":
    run_cli("Train a tiny CortexMesh model and solve a digit-rule prompt.", run)
