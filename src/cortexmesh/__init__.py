"""CortexMesh public API."""

from .config import CortexMeshConfig
from .data import CharTokenizer, SyntheticTaskFactory
from .evaluation import (
    compute_recall_accuracy,
    compute_rule_accuracy,
    compute_token_accuracy,
    cycle_delta_norm,
    evaluate_batch,
    memory_slot_entropy,
)
from .inference import generate_text, recall_memory, solve_rule_task
from .model import CortexMesh
from .trainer import Trainer

__all__ = [
    "CharTokenizer",
    "CortexMesh",
    "CortexMeshConfig",
    "SyntheticTaskFactory",
    "Trainer",
    "compute_recall_accuracy",
    "compute_rule_accuracy",
    "compute_token_accuracy",
    "cycle_delta_norm",
    "evaluate_batch",
    "generate_text",
    "memory_slot_entropy",
    "recall_memory",
    "solve_rule_task",
]
