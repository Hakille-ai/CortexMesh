"""Lightweight character-level baselines for CortexMesh experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class CharNGramBaseline:
    """A deterministic character n-gram next-token baseline.

    The model stores counts for suffix contexts up to ``max_order`` and backs
    off to shorter suffixes when a context has not been observed.
    """

    max_order: int = 4
    _counts: dict[str, Counter[str]] = field(default_factory=dict, init=False)
    _is_fit: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")

    def fit(self, text: str) -> "CharNGramBaseline":
        """Collect next-character counts from ``text``."""

        counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for index, char in enumerate(text):
            max_context = min(self.max_order, index)
            for order in range(max_context + 1):
                context = text[index - order : index] if order else ""
                counts[context][char] += 1

        self._counts = dict(counts)
        self._is_fit = True
        return self

    def predict_next(self, context: str) -> str:
        """Predict the most likely next character for ``context``."""

        self._ensure_fit()
        max_context = min(self.max_order, len(context))
        for order in range(max_context, -1, -1):
            suffix = context[-order:] if order else ""
            if suffix in self._counts:
                return self._most_common(self._counts[suffix])
        return ""

    def generate(self, prompt: str, steps: int) -> str:
        """Append ``steps`` predicted characters to ``prompt``."""

        if steps < 0:
            raise ValueError("steps must be non-negative")

        generated = prompt
        for _ in range(steps):
            next_char = self.predict_next(generated)
            if not next_char:
                break
            generated += next_char
        return generated

    def score_next_token_accuracy(self, text: str, seq_len: int) -> float:
        """Return next-character accuracy using fixed-length left contexts."""

        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")
        self._ensure_fit()

        correct = 0
        total = 0
        for index, expected in enumerate(text):
            if index < seq_len:
                continue
            context = text[index - seq_len : index]
            correct += int(self.predict_next(context) == expected)
            total += 1

        return correct / total if total else 0.0

    def _ensure_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("CharNGramBaseline must be fit before prediction")

    @staticmethod
    def _most_common(counts: Counter[str]) -> str:
        max_count = max(counts.values())
        return min(char for char, count in counts.items() if count == max_count)
