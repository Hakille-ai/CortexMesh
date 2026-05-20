"""Train CortexMesh on a user-provided text corpus from the command line."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import CortexMeshConfig
from .data import CharTokenizer, TextCorpusFactory
from .inference import generate_text
from .model import CortexMesh
from .trainer import Trainer


DEFAULT_TEXT = (
    "CortexMesh learns from a tiny custom corpus. "
    "Concepts compress signals, memory stores traces, and routing mixes them. "
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(tokenizer: CharTokenizer, seq_len: int) -> CortexMesh:
    config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=32,
        concept_dim=24,
        memory_slots=8,
        cycles=2,
        route_hidden_dim=48,
        max_seq_len=seq_len,
    )
    return CortexMesh(config)


def load_text(text: str | None, text_file: str | None) -> tuple[str, str]:
    if text_file is not None:
        path = Path(text_file)
        return path.read_text(encoding="utf-8"), str(path)
    return text if text is not None else DEFAULT_TEXT, "inline"


def run(
    *,
    text: str = DEFAULT_TEXT,
    source_name: str = "inline",
    steps: int = 20,
    batch_size: int = 8,
    seq_len: int = 24,
    seed: int = 13,
    save_dir: str | None = None,
) -> dict[str, Any]:
    """Train a small CortexMesh model on text and return a serializable report."""

    set_seed(seed)
    tokenizer = CharTokenizer.from_text(text)
    model = build_model(tokenizer, seq_len)
    factory = TextCorpusFactory(text, tokenizer, seq_len=seq_len, seed=seed, name=source_name)
    eval_factory = TextCorpusFactory(text, tokenizer, seq_len=seq_len, seed=seed + 1, name=source_name)
    trainer = Trainer(
        model,
        tokenizer,
        lr=4e-3,
        seq_len=seq_len,
        factory=factory,
        eval_factory=eval_factory,
    )
    report = trainer.train_steps(steps=steps, batch_size=batch_size, fixed_batch=False, eval_batches=1)
    prompt = text[: min(8, len(text))] or "cortex"
    sample = generate_text(model, tokenizer, prompt, steps=24, temperature=0.7)

    saved_to = None
    if save_dir is not None:
        save_path = Path(save_dir)
        model.save_pretrained(save_path)
        tokenizer.save_json(save_path / "tokenizer.json")
        saved_to = str(save_path)

    eval_before = report["eval_before"]
    eval_after = report["eval_after"]
    return {
        "source": source_name,
        "source_chars": len(text),
        "vocab_size": tokenizer.vocab_size,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seed": seed,
        "first_loss": report["first_loss"],
        "last_loss": report["last_loss"],
        "eval_before_loss": eval_before["loss"] if eval_before is not None else None,
        "eval_after_loss": eval_after["loss"] if eval_after is not None else None,
        "eval_delta": report["eval_delta"],
        "skipped_steps": report["skipped_steps"],
        "sample": sample,
        "saved_to": saved_to,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CortexMesh on a tiny custom text corpus.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", default=None, help="Inline text corpus. Uses a bundled sample if omitted.")
    input_group.add_argument("--text-file", default=None, help="UTF-8 text corpus file.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--json-output", action="store_true", help="Print the training report as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    text, source_name = load_text(args.text, args.text_file)
    result = run(
        text=text,
        source_name=source_name,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    if args.json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print("CortexMesh text training")
    print(f"source: {result['source']} ({result['source_chars']} chars)")
    print(f"train loss: {result['first_loss']:.4f} -> {result['last_loss']:.4f}")
    if result["eval_before_loss"] is not None and result["eval_after_loss"] is not None:
        print(
            f"eval loss:  {result['eval_before_loss']:.4f} -> "
            f"{result['eval_after_loss']:.4f} ({result['eval_delta']:+.4f})"
        )
    if result["saved_to"] is not None:
        print(f"saved to: {result['saved_to']}")
    print(f"sample: {result['sample']}")


if __name__ == "__main__":
    main()
