"""Minimal reproducible CPU benchmark for CortexMesh."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cortexmesh import CharTokenizer, CortexMesh, CortexMeshConfig, Trainer


@dataclass(frozen=True)
class BenchmarkConfig:
    signal_dim: int
    concept_dim: int
    memory_slots: int
    cycles: int
    route_hidden_dim: int
    max_seq_len: int
    lr: float
    eval_batches: int


CONFIGS: dict[str, BenchmarkConfig] = {
    "tiny": BenchmarkConfig(
        signal_dim=16,
        concept_dim=12,
        memory_slots=4,
        cycles=1,
        route_hidden_dim=24,
        max_seq_len=12,
        lr=5e-3,
        eval_batches=1,
    ),
    "small": BenchmarkConfig(
        signal_dim=32,
        concept_dim=24,
        memory_slots=8,
        cycles=2,
        route_hidden_dim=48,
        max_seq_len=24,
        lr=4e-3,
        eval_batches=2,
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config_name: str) -> tuple[CortexMesh, CharTokenizer, BenchmarkConfig]:
    bench_config = CONFIGS[config_name]
    tokenizer = CharTokenizer()
    model_config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=bench_config.signal_dim,
        concept_dim=bench_config.concept_dim,
        memory_slots=bench_config.memory_slots,
        cycles=bench_config.cycles,
        route_hidden_dim=bench_config.route_hidden_dim,
        max_seq_len=bench_config.max_seq_len,
    )
    return CortexMesh(model_config), tokenizer, bench_config


def _loss_dict(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return None


def run_benchmark(
    *,
    config_name: str = "tiny",
    steps: int = 2,
    batch_size: int = 4,
    seed: int = 13,
    eval_batches: int | None = None,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    torch.set_num_threads(1)
    set_seed(seed)

    model, tokenizer, bench_config = build_model(config_name)
    trainer = Trainer(
        model,
        tokenizer,
        lr=bench_config.lr,
        seq_len=bench_config.max_seq_len,
        device="cpu",
    )

    start = time.perf_counter()
    train_report = trainer.train_steps(
        steps=steps,
        batch_size=batch_size,
        fixed_batch=False,
        eval_batches=0,
    )
    train_seconds = time.perf_counter() - start

    eval_count = bench_config.eval_batches if eval_batches is None else eval_batches
    eval_report = trainer.evaluate(eval_count, batch_size) if eval_count else None

    return {
        "config": config_name,
        "config_details": asdict(bench_config),
        "device": "cpu",
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "train_seconds": train_seconds,
        "train_first_loss": train_report["first_loss"],
        "train_last_loss": train_report["last_loss"],
        "train_last_breakdown": _loss_dict(train_report["last_breakdown"]),
        "skipped_steps": train_report["skipped_steps"],
        "eval_batches": eval_count,
        "eval_loss": eval_report["loss"] if eval_report is not None else None,
        "eval_breakdown": eval_report["breakdown_dict"] if eval_report is not None else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal CortexMesh CPU benchmark.")
    parser.add_argument("--config", choices=sorted(CONFIGS), default="tiny")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Override evaluation batches; use 0 to skip evaluation.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for writing the benchmark result as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    result = run_benchmark(
        config_name=args.config,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_batches=args.eval_batches,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return result


if __name__ == "__main__":
    main()
