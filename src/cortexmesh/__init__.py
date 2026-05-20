"""CortexMesh public API."""

from .config import CortexMeshConfig
from .data import CharTokenizer, SyntheticTaskFactory
from .inference import generate_text, recall_memory, solve_rule_task
from .model import CortexMesh
from .trainer import Trainer

__all__ = [
    "CharTokenizer",
    "CortexMesh",
    "CortexMeshConfig",
    "SyntheticTaskFactory",
    "Trainer",
    "generate_text",
    "recall_memory",
    "solve_rule_task",
]
