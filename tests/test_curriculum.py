from __future__ import annotations

import torch

from cortexmesh import CortexMesh, CortexMeshConfig, Trainer
from cortexmesh.data import CharTokenizer, CurriculumTaskFactory, SyntheticTaskFactory


def test_curriculum_phase_zero_uses_short_text_only_batches() -> None:
    tokenizer = CharTokenizer()
    batch = CurriculumTaskFactory(tokenizer, seq_len=16, seed=31, phase=0).make_batch(
        5,
        fixed_cycle=True,
    )

    assert batch.inputs.shape == (5, 8)
    assert batch.targets.shape == (5, 8)
    assert batch.metadata["task_counts"] == {"text": 5}
    assert batch.metadata["curriculum"]["phase"] == 0
    assert batch.metadata["curriculum"]["seq_len"] == 8
    assert batch.metadata["curriculum"]["task_mix"] == {"text": 1}
    assert not batch.rule_mask.any()
    assert not batch.recall_mask.any()


def test_curriculum_schedule_selects_phase_by_step_and_mixes_tasks() -> None:
    tokenizer = CharTokenizer()
    factory = CurriculumTaskFactory(
        tokenizer,
        seq_len=12,
        seed=41,
        schedule=((0, 0), (3, 1), (9, 2)),
    )

    phase_one = factory.make_batch(6, fixed_cycle=True, step=3)
    phase_two = factory.make_batch(6, fixed_cycle=True, step=9)

    assert phase_one.inputs.shape == (6, 9)
    assert phase_one.metadata["task_counts"] == {"rule": 2, "text": 4}
    assert phase_one.metadata["curriculum"]["phase"] == 1

    assert phase_two.inputs.shape == (6, 12)
    assert phase_two.metadata["task_counts"] == {"memory": 2, "rule": 2, "text": 2}
    assert phase_two.metadata["curriculum"]["phase"] == 2
    assert phase_two.rule_mask.sum().item() == 2
    assert phase_two.recall_mask.sum().item() == 2


def test_curriculum_batches_are_seed_reproducible_with_metadata() -> None:
    tokenizer = CharTokenizer()
    first = CurriculumTaskFactory(tokenizer, seq_len=14, seed=53, phase=2).make_batch(
        6,
        fixed_cycle=False,
        step=25,
    )
    second = CurriculumTaskFactory(tokenizer, seq_len=14, seed=53, phase=2).make_batch(
        6,
        fixed_cycle=False,
        step=25,
    )

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)
    assert first.metadata == second.metadata
    assert first.metadata["curriculum"]["fixed_phase"] is True


def test_curriculum_wraps_synthetic_factory_without_leaving_seq_len_mutated() -> None:
    tokenizer = CharTokenizer()
    synthetic = SyntheticTaskFactory(tokenizer, seq_len=20, seed=67)
    factory = CurriculumTaskFactory(
        tokenizer,
        seq_len=20,
        seed=67,
        phase=0,
        synthetic_factory=synthetic,
    )

    batch = factory.make_batch(3, fixed_cycle=True)

    assert batch.inputs.shape == (3, 10)
    assert synthetic.seq_len == 20


def test_trainer_can_use_curriculum_factory_schedule() -> None:
    torch.manual_seed(71)
    tokenizer = CharTokenizer()
    config = CortexMeshConfig(
        vocab_size=tokenizer.vocab_size,
        signal_dim=16,
        concept_dim=12,
        memory_slots=4,
        cycles=1,
        route_hidden_dim=24,
        max_seq_len=12,
    )
    model = CortexMesh(config)
    factory = CurriculumTaskFactory(tokenizer, seq_len=config.max_seq_len, seed=71, schedule=((0, 0), (1, 2)))
    eval_factory = CurriculumTaskFactory(tokenizer, seq_len=config.max_seq_len, seed=72, phase=2)
    trainer = Trainer(
        model,
        tokenizer,
        lr=5e-3,
        seq_len=config.max_seq_len,
        factory=factory,
        eval_factory=eval_factory,
    )

    report = trainer.train_steps(steps=2, batch_size=3, fixed_batch=False, eval_batches=1)
    eval_report = trainer.evaluate(batches=1, batch_size=3, include_metrics=True)

    assert report["skipped_steps"] == 0
    assert "metrics" in eval_report
    assert set(eval_report["metrics"]) == {
        "token_accuracy",
        "rule_accuracy",
        "recall_accuracy",
        "memory_slot_entropy",
        "cycle_delta_norm",
    }
