"""Configuration for CortexMesh."""

from __future__ import annotations

from dataclasses import dataclass


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
