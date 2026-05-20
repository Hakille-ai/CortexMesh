"""CortexMesh public API."""

from .baselines import CharNGramBaseline
from .config import CortexMeshConfig
from .data import CharTokenizer, CurriculumTaskFactory, SyntheticTaskFactory, TextCorpusFactory
from .evaluation import (
    compute_recall_accuracy,
    compute_rule_accuracy,
    compute_token_accuracy,
    cycle_delta_norm,
    evaluate_batch,
    memory_slot_entropy,
)
from .inference import generate_text, recall_memory, solve_rule_task
from .inspection import summarize_output, summarize_prompt, top_memory_reads
from .model import CortexMesh
from .trainer import Trainer

__all__ = [
    "CharTokenizer",
    "CharNGramBaseline",
    "CortexMesh",
    "CortexMeshConfig",
    "CurriculumTaskFactory",
    "SyntheticTaskFactory",
    "TextCorpusFactory",
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
    "summarize_output",
    "summarize_prompt",
    "top_memory_reads",
]
