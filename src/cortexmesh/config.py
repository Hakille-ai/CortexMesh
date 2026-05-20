"""Configuration for CortexMesh."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CortexMeshConfig:
    """Shape and behavior settings for a CortexMesh model."""

    vocab_size: int
    signal_dim: int = 64
    concept_dim: int = 48
    memory_slots: int = 16
    cycles: int = 3
    route_hidden_dim: int = 96
    max_seq_len: int = 64
    rule_classes: int = 4
    graph_mix_layers: int = 0
    graph_radius: int = 1

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable representation of this config."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CortexMeshConfig":
        """Build a config from a dictionary."""
        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - field_names)
        if unknown:
            raise ValueError(f"Unknown config field(s): {', '.join(unknown)}")

        config = cls(**{name: data[name] for name in field_names if name in data})
        config.validate()
        return config

    def save_json(self, path: str | Path) -> None:
        """Save this config as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        path.write_text(contents, encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "CortexMeshConfig":
        """Load a config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config JSON must contain an object")
        return cls.from_dict(data)

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.signal_dim < 4:
            raise ValueError("signal_dim must be at least 4")
        if self.concept_dim < 4:
            raise ValueError("concept_dim must be at least 4")
        if self.memory_slots < 2:
            raise ValueError("memory_slots must be at least 2")
        if self.cycles < 1:
            raise ValueError("cycles must be at least 1")
        if self.max_seq_len < 4:
            raise ValueError("max_seq_len must be at least 4")
        if self.rule_classes < 4:
            raise ValueError("rule_classes must be at least 4")
        if self.graph_mix_layers < 0:
            raise ValueError("graph_mix_layers must be non-negative")
        if self.graph_radius < 1:
            raise ValueError("graph_radius must be at least 1")
