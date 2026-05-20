from __future__ import annotations

import torch

from cortexmesh import (
    CharTokenizer,
    CortexMesh,
    CortexMeshConfig,
)
from cortexmesh.inspection import (
    summarize_output,
    summarize_prompt,
    top_memory_reads,
)


def tiny_model() -> tuple[CortexMesh, CharTokenizer]:
    tokenizer = CharTokenizer()
    config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=16,
        concept_dim=12,
        memory_slots=4,
        cycles=2,
        route_hidden_dim=24,
        max_seq_len=10,
    )
    return CortexMesh(config), tokenizer


def test_summarize_output_reports_memory_and_concept_fields() -> None:
    model, tokenizer = tiny_model()
    tokens = torch.randint(0, tokenizer.vocab_size, (2, 7))

    output = model(tokens)
    report = summarize_output(output)

    assert report["read_weights_shape"] == (7, 4)
    assert report["memory_shape"] == (4, 12)
    assert report["concepts_shape"] == (7, 12)
    assert 0.999 <= report["read_weight_row_sum_min"] <= 1.001
    assert 0.999 <= report["read_weight_row_sum_max"] <= 1.001
    assert report["memory_l2_max"] > 0
    assert report["concept_l2_max"] > 0
    assert "cycle_delta_l2_mean" in report


def test_top_memory_reads_returns_slots_per_token() -> None:
    output = {
        "read_weights": torch.tensor(
            [
                [
                    [0.1, 0.7, 0.2],
                    [0.8, 0.1, 0.1],
                ]
            ]
        )
    }

    reads = top_memory_reads(output, tokens="ab", top_k=2)

    assert reads[0]["token"] == "a"
    assert reads[0]["slot"] == 1
    assert reads[0]["weight"] == 0.7
    assert reads[0]["top_slots"] == [{"slot": 1, "weight": 0.7}, {"slot": 2, "weight": 0.2}]
    assert reads[1]["slot"] == 0


def test_summarize_prompt_runs_on_cpu_and_includes_top_reads() -> None:
    model, tokenizer = tiny_model()

    report = summarize_prompt(model, tokenizer, "a=3;?a=")

    assert report["prompt"] == "a=3;?a="
    assert report["token_count"] == 7
    assert report["read_weights_shape"] == (7, 4)
    assert report["memory_shape"] == (4, 12)
    assert report["concepts_shape"] == (7, 12)
    assert len(report["top_reads"]) == report["token_count"]
