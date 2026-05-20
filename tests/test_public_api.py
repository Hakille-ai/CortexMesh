from __future__ import annotations

from cortexmesh import (
    CharNGramBaseline,
    CurriculumTaskFactory,
    summarize_output,
    summarize_prompt,
    top_memory_reads,
)


def test_public_research_helpers_are_exported() -> None:
    assert CharNGramBaseline(max_order=1).max_order == 1
    assert CurriculumTaskFactory().phase_for_step(0) == 0
    assert callable(summarize_output)
    assert callable(summarize_prompt)
    assert callable(top_memory_reads)
