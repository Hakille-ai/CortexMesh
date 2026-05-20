from __future__ import annotations

import random
from typing import Callable, TypeVar

import numpy as np
import torch

from cortexmesh import CortexMesh, CortexMeshConfig, CharTokenizer, Trainer

T = TypeVar("T")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_small_trained_model(
    *,
    steps: int,
    batch_size: int = 8,
    seed: int = 13,
) -> tuple[CortexMesh, CharTokenizer, dict[str, object]]:
    set_seed(seed)
    tokenizer = CharTokenizer()
    config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=32,
        concept_dim=24,
        memory_slots=8,
        cycles=2,
        route_hidden_dim=48,
        max_seq_len=24,
    )
    model = CortexMesh(config)
    trainer = Trainer(model, tokenizer, lr=4e-3, seq_len=config.max_seq_len)
    report = trainer.train_steps(steps=steps, batch_size=batch_size)
    return model, tokenizer, report


def run_cli(
    title: str,
    runner: Callable[[int, int], T],
    *,
    default_steps: int = 20,
    default_seed: int = 13,
) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--seed", type=int, default=default_seed)
    args = parser.parse_args()
    print(runner(args.steps, args.seed))
