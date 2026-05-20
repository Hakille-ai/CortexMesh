"""CortexMesh model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import CortexMeshConfig
from .modules import ConceptField, ConceptGraphMixer, MemoryLattice, PredictionHead, RouteMixer, SignalEncoder


class CortexMesh(nn.Module):
    """Small experimental model with concept routing and external memory."""

    config_filename = "config.json"
    weights_filename = "pytorch_model.bin"

    def __init__(self, config: CortexMeshConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = SignalEncoder(config.vocab_size, config.signal_dim)
        self.concepts = ConceptField(config.signal_dim, config.concept_dim)
        self.memory = MemoryLattice(config.concept_dim, config.memory_slots)
        self.router = RouteMixer(config.concept_dim, config.route_hidden_dim)
        self.graph_mixer = (
            ConceptGraphMixer(config.concept_dim, config.route_hidden_dim, radius=config.graph_radius)
            if config.graph_mix_layers
            else None
        )
        self.head = PredictionHead(
            config.concept_dim,
            config.vocab_size,
            config.rule_classes,
            config.route_hidden_dim,
        )
        self._missing_ok_state_keys = self._optional_state_keys()

    def save_pretrained(self, path: str | Path) -> None:
        """Save config and model weights to a local directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save_json(path / self.config_filename)
        torch.save(self.state_dict(), path / self.weights_filename)

    @classmethod
    def from_pretrained(cls, path: str | Path, map_location: str | torch.device | None = "cpu") -> "CortexMesh":
        """Load a CortexMesh model from a local directory."""
        path = Path(path)
        config = CortexMeshConfig.load_json(path / cls.config_filename)
        model = cls(config)
        state_dict = _load_state_dict(path / cls.weights_filename, map_location=map_location)
        model.load_state_dict(state_dict)
        return model

    def forward(self, tokens: torch.Tensor, return_internal: bool = True) -> dict[str, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")

        signals = self.encoder(tokens)
        concepts, initial_reconstruction = self.concepts(signals)
        cycle_states = []
        graph_states = []
        readout = torch.zeros_like(concepts)
        memory_state = self.memory.seed.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        read_weights = torch.zeros(
            tokens.shape[0],
            tokens.shape[1],
            self.config.memory_slots,
            device=tokens.device,
            dtype=signals.dtype,
        )

        for _ in range(self.config.cycles):
            readout, memory_state, read_weights = self.memory(concepts)
            concepts = self.router(concepts, readout)
            if self.graph_mixer is not None:
                for _ in range(self.config.graph_mix_layers):
                    concepts = self.graph_mixer(concepts)
                    graph_states.append(concepts)
            cycle_states.append(concepts)

        reconstruction = self.concepts.decode(concepts)
        logits, rule_logits, recall_logits, summary = self.head(concepts, memory_state)
        output = {
            "logits": logits,
            "rule_logits": rule_logits,
            "recall_logits": recall_logits,
            "reconstruction": reconstruction,
            "initial_reconstruction": initial_reconstruction,
            "summary": summary,
        }
        if return_internal:
            output.update(
                {
                    "signals": signals,
                    "concepts": concepts,
                    "memory": memory_state,
                    "readout": readout,
                    "read_weights": read_weights,
                    "cycle_states": torch.stack(cycle_states, dim=0),
                }
            )
            if graph_states:
                output["graph_states"] = torch.stack(graph_states, dim=0)
        return output

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        """Load weights while tolerating newly optional architecture modules.

        Older CortexMesh checkpoints do not include optional graph-mixer weights.
        When the current config enables that mixer, those missing keys are
        initialized from the current module parameters.
        """

        if strict and self._missing_ok_state_keys:
            result = super().load_state_dict(state_dict, strict=False, assign=assign)
            unexpected = list(result.unexpected_keys)
            missing = [key for key in result.missing_keys if key not in self._missing_ok_state_keys]
            if missing or unexpected:
                raise RuntimeError(
                    "Error(s) in loading state_dict for CortexMesh: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            return result
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _optional_state_keys(self) -> set[str]:
        if self.graph_mixer is None:
            return set()
        return {f"graph_mixer.{key}" for key in self.graph_mixer.state_dict()}


def _load_state_dict(path: Path, map_location: str | torch.device | None) -> dict[str, Any]:
    try:
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=map_location)
    if not isinstance(state_dict, dict):
        raise ValueError("Model weights file must contain a state_dict")
    return state_dict
