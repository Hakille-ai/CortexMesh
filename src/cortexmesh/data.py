"""Tokenization, synthetic tasks, and custom text-corpus utilities."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, overload

import torch


DEFAULT_ALPHABET = " abcdefghijklmnopqrstuvwxyz0123456789.,;:=?+-*/\n"


class CharTokenizer:
    """Tiny character tokenizer for local synthetic experiments."""

    def __init__(self, alphabet: str = DEFAULT_ALPHABET) -> None:
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must not contain duplicate characters")
        if " " not in alphabet:
            raise ValueError("alphabet must contain a space for unknown characters")
        self.alphabet = alphabet
        self.stoi = {char: idx for idx, char in enumerate(alphabet)}
        self.itos = {idx: char for char, idx in self.stoi.items()}
        self.unknown_id = self.stoi[" "]

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def encode(self, text: str, pad_to: int | None = None) -> list[int]:
        ids = [self.stoi.get(char.lower(), self.unknown_id) for char in text]
        if pad_to is not None:
            if len(ids) > pad_to:
                ids = ids[:pad_to]
            ids = ids + [self.unknown_id] * (pad_to - len(ids))
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        return "".join(self.itos.get(int(idx), " ") for idx in ids)

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable tokenizer description."""

        return {"alphabet": self.alphabet}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CharTokenizer":
        """Build a tokenizer from serialized data."""

        alphabet = data.get("alphabet")
        if not isinstance(alphabet, str):
            raise ValueError("Tokenizer data must include an alphabet string")
        return cls(alphabet=alphabet)

    def save_json(self, path: str | Path) -> None:
        """Save this tokenizer to JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        path.write_text(contents, encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "CharTokenizer":
        """Load a tokenizer from JSON."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Tokenizer JSON must contain an object")
        return cls.from_dict(data)

    @classmethod
    def from_text(
        cls,
        text: str,
        extra_alphabet: str = DEFAULT_ALPHABET,
    ) -> "CharTokenizer":
        """Create a tokenizer that covers the default alphabet plus text chars."""

        alphabet = _unique_chars(extra_alphabet + text.lower())
        return cls(alphabet=alphabet)


@dataclass
class SyntheticBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    rule_labels: torch.Tensor
    rule_mask: torch.Tensor
    recall_targets: torch.Tensor
    recall_mask: torch.Tensor
    metadata: dict[str, object] = field(default_factory=dict)

    @overload
    def as_dict(self, include_metadata: Literal[False] = False) -> dict[str, torch.Tensor]:
        ...

    @overload
    def as_dict(self, include_metadata: Literal[True]) -> dict[str, object]:
        ...

    def as_dict(self, include_metadata: bool = False) -> dict[str, object]:
        batch: dict[str, object] = {
            "inputs": self.inputs,
            "targets": self.targets,
            "rule_labels": self.rule_labels,
            "rule_mask": self.rule_mask,
            "recall_targets": self.recall_targets,
            "recall_mask": self.recall_mask,
        }
        if include_metadata:
            batch["metadata"] = self.metadata
        return batch


@dataclass(frozen=True)
class _SyntheticExample:
    text: str
    rule_label: int
    recall_target: int
    is_rule: bool
    is_memory: bool
    task_type: str
    variant: str
    query_key: str | None = None
    answer: str | None = None


class SyntheticTaskFactory:
    """Generate tiny text, rule, and memory tasks with no external dataset."""

    def __init__(
        self,
        tokenizer: CharTokenizer | None = None,
        seq_len: int = 32,
        seed: int = 7,
    ) -> None:
        self.tokenizer = tokenizer or CharTokenizer()
        self.seq_len = seq_len
        self.seed = seed
        self.rng = random.Random(seed)
        self.text_bank = [
            "cortex mesh learns compact memory. ",
            "small models can route signals. ",
            "concepts compress text and rules. ",
            "memory helps recall useful facts. ",
            "routing blends local context with state. ",
            "latent concepts keep signals organized. ",
            "tiny agents can share sparse memories. ",
            "short traces teach sequence prediction. ",
        ]
        self.rule_names = {
            0: "increment",
            1: "skip_two",
            2: "alternate_pair",
            3: "cycle_three",
        }

    def _fit(self, text: str) -> str:
        need = self.seq_len + 1
        if not text:
            text = " "
        while len(text) < need:
            text += text
        return text[:need]

    def _text_example(self) -> _SyntheticExample:
        base = self.rng.choice(self.text_bank)
        variants = (
            base,
            base.replace(".", "?"),
            f"{base.strip()}; {self.rng.choice(self.text_bank)}",
            f"{self.rng.choice(['note', 'trace', 'memo'])}: {base}",
        )
        variant_index = self.rng.randrange(len(variants))
        doubled = variants[variant_index] * 4
        start = self.rng.randrange(max(1, len(doubled) - self.seq_len - 1))
        text = self._fit(doubled[start:])
        return _SyntheticExample(
            text=text,
            rule_label=0,
            recall_target=self.tokenizer.encode(text[0])[0],
            is_rule=False,
            is_memory=False,
            task_type="text",
            variant=f"text_{variant_index}",
        )

    def text_sequence(self) -> tuple[str, int, int, bool, bool]:
        example = self._text_example()
        return (
            example.text,
            example.rule_label,
            example.recall_target,
            example.is_rule,
            example.is_memory,
        )

    def _rule_example(self) -> _SyntheticExample:
        rule = self.rng.randrange(4)
        digits = "0123456789"
        if rule == 0:
            start = self.rng.randrange(10)
            step = self.rng.choice([1, 3])
            seq = "".join(digits[(start + i * step) % 10] for i in range(self.seq_len + 1))
        elif rule == 1:
            start = self.rng.randrange(10)
            step = self.rng.choice([2, 4])
            seq = "".join(digits[(start + i * step) % 10] for i in range(self.seq_len + 1))
        elif rule == 2:
            a = self.rng.randrange(10)
            b = (a + self.rng.randrange(1, 10)) % 10
            offset = self.rng.randrange(2)
            seq = "".join(str(a if (i + offset) % 2 == 0 else b) for i in range(self.seq_len + 1))
        else:
            a = self.rng.randrange(10)
            jump = self.rng.choice([2, 3, 5])
            b = (a + jump) % 10
            c = (a + jump * 2) % 10
            cycle = [a, b, c]
            offset = self.rng.randrange(3)
            seq = "".join(str(cycle[(i + offset) % 3]) for i in range(self.seq_len + 1))
        return _SyntheticExample(
            text=seq,
            rule_label=rule,
            recall_target=self.tokenizer.encode(seq[-1])[0],
            is_rule=True,
            is_memory=False,
            task_type="rule",
            variant=self.rule_names[rule],
        )

    def rule_sequence(self) -> tuple[str, int, int, bool, bool]:
        example = self._rule_example()
        return (
            example.text,
            example.rule_label,
            example.recall_target,
            example.is_rule,
            example.is_memory,
        )

    def _memory_example(self) -> _SyntheticExample:
        key_sets = (
            ["a", "b", "c", "d"],
            ["m", "n", "p", "q"],
            ["x", "y", "z", "r"],
        )
        keys = list(self.rng.choice(key_sets))
        self.rng.shuffle(keys)
        values = [str(self.rng.randrange(10)) for _ in keys]
        query_index = self.rng.randrange(len(keys))
        separator = self.rng.choice([";", ","])
        pairs = separator.join(f"{key}={value}" for key, value in zip(keys, values))
        answer = values[query_index]
        query_key = keys[query_index]
        query = self.rng.choice([f"?{query_key}={answer}", f"{query_key}?{answer}"])
        text = self._fit(f"{pairs};{query}")
        return _SyntheticExample(
            text=text,
            rule_label=3,
            recall_target=self.tokenizer.encode(answer)[0],
            is_rule=False,
            is_memory=True,
            task_type="memory",
            variant=f"{len(keys)}_pairs",
            query_key=query_key,
            answer=answer,
        )

    def memory_sequence(self) -> tuple[str, int, int, bool, bool]:
        example = self._memory_example()
        return (
            example.text,
            example.rule_label,
            example.recall_target,
            example.is_rule,
            example.is_memory,
        )

    def make_batch(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        fixed_cycle: bool = False,
    ) -> SyntheticBatch:
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        rule_labels: list[int] = []
        rule_mask: list[bool] = []
        recall_targets: list[int] = []
        recall_mask: list[bool] = []
        task_types: list[str] = []
        variants: list[str] = []
        query_keys: list[str | None] = []
        answers: list[str | None] = []

        for idx in range(batch_size):
            choice = idx % 3 if fixed_cycle else self.rng.randrange(3)
            if choice == 0:
                example = self._text_example()
            elif choice == 1:
                example = self._rule_example()
            else:
                example = self._memory_example()

            ids = self.tokenizer.encode(example.text, pad_to=self.seq_len + 1)
            inputs.append(ids[:-1])
            targets.append(ids[1:])
            rule_labels.append(example.rule_label)
            rule_mask.append(example.is_rule)
            recall_targets.append(example.recall_target)
            recall_mask.append(example.is_memory)
            task_types.append(example.task_type)
            variants.append(example.variant)
            query_keys.append(example.query_key)
            answers.append(example.answer)

        tensor_device = torch.device(device) if device is not None else None
        task_counts = {task: task_types.count(task) for task in sorted(set(task_types))}
        return SyntheticBatch(
            inputs=torch.tensor(inputs, dtype=torch.long, device=tensor_device),
            targets=torch.tensor(targets, dtype=torch.long, device=tensor_device),
            rule_labels=torch.tensor(rule_labels, dtype=torch.long, device=tensor_device),
            rule_mask=torch.tensor(rule_mask, dtype=torch.bool, device=tensor_device),
            recall_targets=torch.tensor(recall_targets, dtype=torch.long, device=tensor_device),
            recall_mask=torch.tensor(recall_mask, dtype=torch.bool, device=tensor_device),
            metadata={
                "seed": self.seed,
                "seq_len": self.seq_len,
                "batch_size": batch_size,
                "fixed_cycle": fixed_cycle,
                "task_types": task_types,
                "task_counts": task_counts,
                "variants": variants,
                "memory_query_keys": query_keys,
                "memory_answers": answers,
            },
        )


class TextCorpusFactory:
    """Create next-character batches from a user-provided text corpus."""

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer | None = None,
        seq_len: int = 32,
        seed: int = 7,
        name: str = "inline",
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be at least 1")
        if not text:
            raise ValueError("text corpus must not be empty")

        self.raw_text = text
        self.tokenizer = tokenizer or CharTokenizer.from_text(text)
        self.seq_len = seq_len
        self.seed = seed
        self.name = name
        self.rng = random.Random(seed)
        self.ids = self.tokenizer.encode(text)
        while len(self.ids) < self.seq_len + 1:
            self.ids.extend(self.ids or [self.tokenizer.unknown_id])

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        tokenizer: CharTokenizer | None = None,
        seq_len: int = 32,
        seed: int = 7,
        encoding: str = "utf-8",
    ) -> "TextCorpusFactory":
        """Load a text corpus from a local file."""

        path = Path(path)
        return cls(
            path.read_text(encoding=encoding),
            tokenizer=tokenizer,
            seq_len=seq_len,
            seed=seed,
            name=str(path),
        )

    @property
    def max_start(self) -> int:
        return max(0, len(self.ids) - self.seq_len - 1)

    def make_batch(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        fixed_cycle: bool = False,
    ) -> SyntheticBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        starts: list[int] = []
        inputs: list[list[int]] = []
        targets: list[list[int]] = []
        span = self.max_start + 1
        stride = max(1, self.seq_len // 2)
        for idx in range(batch_size):
            if fixed_cycle:
                start = (idx * stride) % span
            else:
                start = self.rng.randrange(span)
            window = self.ids[start : start + self.seq_len + 1]
            inputs.append(window[:-1])
            targets.append(window[1:])
            starts.append(start)

        tensor_device = torch.device(device) if device is not None else None
        zeros = torch.zeros(batch_size, dtype=torch.long, device=tensor_device)
        false_mask = torch.zeros(batch_size, dtype=torch.bool, device=tensor_device)
        return SyntheticBatch(
            inputs=torch.tensor(inputs, dtype=torch.long, device=tensor_device),
            targets=torch.tensor(targets, dtype=torch.long, device=tensor_device),
            rule_labels=zeros.clone(),
            rule_mask=false_mask.clone(),
            recall_targets=zeros.clone(),
            recall_mask=false_mask.clone(),
            metadata={
                "seed": self.seed,
                "seq_len": self.seq_len,
                "batch_size": batch_size,
                "fixed_cycle": fixed_cycle,
                "task_types": ["text_corpus"] * batch_size,
                "task_counts": {"text_corpus": batch_size},
                "source_name": self.name,
                "source_chars": len(self.raw_text),
                "source_tokens": len(self.ids),
                "offsets": starts,
            },
        )


def _unique_chars(text: str) -> str:
    seen: set[str] = set()
    chars: list[str] = []
    for char in text:
        if char not in seen:
            chars.append(char)
            seen.add(char)
    return "".join(chars)
