"""Command-line demo for CortexMesh."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from .config import CortexMeshConfig
from .data import CharTokenizer
from .inference import generate_text, recall_memory, solve_rule_task
from .model import CortexMesh
from .trainer import Trainer


def build_demo_model() -> tuple[CortexMesh, CharTokenizer]:
    tokenizer = CharTokenizer()
    config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=48,
        concept_dim=40,
        memory_slots=12,
        cycles=3,
        route_hidden_dim=72,
        max_seq_len=32,
    )
    return CortexMesh(config), tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and sample CortexMesh v0.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--generate-steps", type=int, default=24)
    parser.add_argument("--prompt", default="cortex ")
    parser.add_argument("--rule-prompt", default="02468")
    parser.add_argument("--memory-prompt", default="a=3;b=7;c=1;?b=")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = build_demo_model()
    trainer = Trainer(model, tokenizer, lr=args.lr, seq_len=model.config.max_seq_len)
    report = trainer.train_steps(
        steps=args.steps,
        batch_size=args.batch_size,
        fixed_batch=args.fixed_batch,
        eval_batches=args.eval_batches,
    )

    print("CortexMesh v0 demo")
    print(f"train loss: {report['first_loss']:.4f} -> {report['last_loss']:.4f}")
    if report["eval_before"] is not None and report["eval_after"] is not None:
        before = report["eval_before"]["breakdown"]
        after = report["eval_after"]["breakdown"]
        print(f"eval loss:  {before.total:.4f} -> {after.total:.4f} ({report['eval_delta']:+.4f})")
        print(
            "after breakdown:"
            f" token={after.token:.4f}"
            f" recon={after.reconstruction:.4f}"
            f" rule={after.rule:.4f}"
            f" recall={after.recall:.4f}"
        )
    if report["skipped_steps"]:
        print(f"skipped non-finite steps: {report['skipped_steps']}")

    print(
        "text:",
        generate_text(
            model,
            tokenizer,
            args.prompt,
            steps=args.generate_steps,
            temperature=args.temperature,
        ),
    )
    print("rule:", solve_rule_task(model, tokenizer, args.rule_prompt))
    print("memory:", recall_memory(model, tokenizer, args.memory_prompt))


if __name__ == "__main__":
    main()
