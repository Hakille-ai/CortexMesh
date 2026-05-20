"""Custom CortexMesh building blocks.

These modules intentionally avoid transformer, recurrent, and convolutional
layers. Routing is performed with learned gates and external memory slots.
"""

from __future__ import annotations

import torch
from torch import nn


class SignalEncoder(nn.Module):
    """Convert token ids into continuous signals plus simple temporal traits."""

    def __init__(self, vocab_size: int, signal_dim: int) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, signal_dim)
        self.time_project = nn.Linear(4, signal_dim, bias=False)
        self.value_gate = nn.Linear(signal_dim, signal_dim)
        self.norm = nn.LayerNorm(signal_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")

        batch, length = tokens.shape
        token_signal = self.token(tokens)
        if length == 1:
            progress = torch.zeros(1, device=tokens.device, dtype=token_signal.dtype)
        else:
            progress = torch.linspace(
                0.0,
                1.0,
                length,
                device=tokens.device,
                dtype=token_signal.dtype,
            )
        age = 1.0 - progress
        center_distance = (progress - 0.5).pow(2)
        parity = torch.where(
            torch.arange(length, device=tokens.device) % 2 == 0,
            torch.ones(length, device=tokens.device, dtype=token_signal.dtype),
            -torch.ones(length, device=tokens.device, dtype=token_signal.dtype),
        )
        traits = torch.stack((progress, age, center_distance, parity), dim=-1)
        time_signal = self.time_project(traits).unsqueeze(0).expand(batch, -1, -1)
        gate = torch.sigmoid(self.value_gate(token_signal))
        return self.norm(token_signal + time_signal + token_signal * gate)


class ConceptField(nn.Module):
    """Compress signals into concepts and reconstruct them."""

    def __init__(self, signal_dim: int, concept_dim: int) -> None:
        super().__init__()
        self.compress = nn.Linear(signal_dim, concept_dim)
        self.sieve = nn.Linear(signal_dim, concept_dim)
        self.expand = nn.Linear(concept_dim, signal_dim)
        self.norm = nn.LayerNorm(concept_dim)

    def encode(self, signals: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.compress(signals))
        gate = torch.sigmoid(self.sieve(signals))
        return self.norm(raw + raw * gate)

    def decode(self, concepts: torch.Tensor) -> torch.Tensor:
        return self.expand(concepts)

    def forward(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        concepts = self.encode(signals)
        reconstruction = self.decode(concepts)
        return concepts, reconstruction


class MemoryLattice(nn.Module):
    """External differentiable memory addressed by learned gates."""

    def __init__(self, concept_dim: int, memory_slots: int) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(memory_slots, concept_dim) * 0.02)
        self.write_gate = nn.Linear(concept_dim, memory_slots)
        self.write_value = nn.Linear(concept_dim, concept_dim)
        self.write_mix = nn.Linear(concept_dim, memory_slots)
        self.read_gate = nn.Linear(concept_dim, memory_slots)
        self.memory_norm = nn.LayerNorm(concept_dim)
        self.read_norm = nn.LayerNorm(concept_dim)

    def write(self, concepts: torch.Tensor) -> torch.Tensor:
        batch = concepts.shape[0]
        base = self.seed.unsqueeze(0).expand(batch, -1, -1)
        gates = torch.sigmoid(self.write_gate(concepts))
        values = torch.tanh(self.write_value(concepts))
        denom = gates.sum(dim=1).clamp_min(1e-5).unsqueeze(-1)
        update = torch.einsum("btm,btd->bmd", gates, values) / denom
        mix = torch.sigmoid(self.write_mix(concepts.mean(dim=1))).unsqueeze(-1)
        return self.memory_norm(base * (1.0 - mix) + update * mix)

    def read(self, concepts: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.read_gate(concepts))
        weights = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        readout = torch.einsum("btm,bmd->btd", weights, memory)
        return self.read_norm(readout), weights

    def forward(self, concepts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = self.write(concepts)
        readout, weights = self.read(concepts, memory)
        return readout, memory, weights


class RouteMixer(nn.Module):
    """Mix local concepts, memory readouts, and whole-sequence statistics."""

    def __init__(self, concept_dim: int, route_hidden_dim: int) -> None:
        super().__init__()
        self.global_mix = nn.Linear(concept_dim * 2, concept_dim)
        self.candidate = nn.Sequential(
            nn.Linear(concept_dim * 3, route_hidden_dim),
            nn.Tanh(),
            nn.Linear(route_hidden_dim, concept_dim),
        )
        self.gate = nn.Linear(concept_dim * 3, concept_dim)
        self.norm = nn.LayerNorm(concept_dim)

    def forward(self, concepts: torch.Tensor, readout: torch.Tensor) -> torch.Tensor:
        mean = concepts.mean(dim=1)
        spread = (concepts - mean.unsqueeze(1)).pow(2).mean(dim=1).sqrt()
        global_trait = torch.tanh(self.global_mix(torch.cat((mean, spread), dim=-1)))
        global_trait = global_trait.unsqueeze(1).expand_as(concepts)
        combined = torch.cat((concepts, readout, global_trait), dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        candidate = torch.tanh(self.candidate(combined))
        return self.norm(concepts * (1.0 - gate) + candidate * gate)


class PredictionHead(nn.Module):
    """Produce token, rule, and memory-recall predictions."""

    def __init__(
        self,
        concept_dim: int,
        vocab_size: int,
        rule_classes: int,
        route_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.token_head = nn.Linear(concept_dim, vocab_size)
        self.summary_project = nn.Sequential(
            nn.Linear(concept_dim * 2, route_hidden_dim),
            nn.Tanh(),
            nn.Linear(route_hidden_dim, concept_dim),
            nn.Tanh(),
        )
        self.rule_head = nn.Linear(concept_dim, rule_classes)
        self.recall_head = nn.Linear(concept_dim, vocab_size)

    def forward(
        self,
        concepts: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.token_head(concepts)
        summary = torch.cat((concepts.mean(dim=1), memory.mean(dim=1)), dim=-1)
        summary_concept = self.summary_project(summary)
        rule_logits = self.rule_head(summary_concept)
        recall_logits = self.recall_head(summary_concept)
        return logits, rule_logits, recall_logits, summary_concept
