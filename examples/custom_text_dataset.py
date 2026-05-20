from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cortexmesh import CortexMesh, CortexMeshConfig, CharTokenizer, TextCorpusFactory, Trainer, generate_text


DEFAULT_TEXT = (
    "CortexMesh learns from a tiny custom corpus. "
    "Concepts compress signals, memory stores traces, and routing mixes them. "
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_text(args: argparse.Namespace) -> tuple[str, str]:
    if args.text_file is not None:
        path = Path(args.text_file)
        return path.read_text(encoding=args.encoding), str(path)
    return args.text, "inline"


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


def run(
    *,
    text: str = DEFAULT_TEXT,
    source_name: str = "inline",
    steps: int = 20,
    batch_size: int = 8,
    seq_len: int = 24,
    seed: int = 13,
    prompt: str = "cortex",
    save_dir: str | None = None,
) -> dict[str, object]:
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
    sample = generate_text(model, tokenizer, prompt, steps=24, temperature=0.7)

    saved_to = None
    if save_dir is not None:
        saved_to = str(Path(save_dir))
        model.save_pretrained(saved_to)
        tokenizer.save_json(Path(saved_to) / "tokenizer.json")

    return {
        "source": source_name,
        "vocab_size": tokenizer.vocab_size,
        "steps": steps,
        "train_loss": [report["first_loss"], report["last_loss"]],
        "eval_loss": report["eval_after"]["loss"] if report["eval_after"] is not None else None,
        "sample": sample,
        "saved_to": saved_to,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CortexMesh on a tiny custom text corpus.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", default=None)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prompt", default="cortex")
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args(argv)

    text, source_name = load_text(args)
    print(
        run(
            text=text,
            source_name=source_name,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            seed=args.seed,
            prompt=args.prompt,
            save_dir=args.save_dir,
        )
    )


if __name__ == "__main__":
    main()
