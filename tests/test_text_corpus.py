from __future__ import annotations

import torch

from cortexmesh import CortexMesh, CortexMeshConfig, CharTokenizer, TextCorpusFactory, Trainer


def tiny_model(tokenizer: CharTokenizer) -> CortexMesh:
    return CortexMesh(
        CortexMeshConfig(
            vocab_size=tokenizer.vocab_size,
            signal_dim=24,
            concept_dim=16,
            memory_slots=6,
            cycles=2,
            route_hidden_dim=32,
            max_seq_len=16,
        )
    )


def test_char_tokenizer_json_roundtrip(tmp_path) -> None:
    tokenizer = CharTokenizer.from_text("CortexMesh: custom TEXT #1\n")
    path = tmp_path / "tokenizer.json"
    tokenizer.save_json(path)

    loaded = CharTokenizer.load_json(path)

    assert loaded.alphabet == tokenizer.alphabet
    assert loaded.encode("CUSTOM #1") == tokenizer.encode("custom #1")
    assert loaded.decode(loaded.encode("mesh")) == "mesh"


def test_text_corpus_factory_is_reproducible_with_metadata() -> None:
    text = "cortex mesh can train on a user text corpus. " * 3
    tokenizer = CharTokenizer.from_text(text)
    first = TextCorpusFactory(text, tokenizer, seq_len=12, seed=5).make_batch(4, fixed_cycle=False)
    second = TextCorpusFactory(text, tokenizer, seq_len=12, seed=5).make_batch(4, fixed_cycle=False)

    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)
    assert first.metadata["task_counts"] == {"text_corpus": 4}
    assert first.metadata["offsets"] == second.metadata["offsets"]


def test_trainer_accepts_text_corpus_factory() -> None:
    torch.manual_seed(9)
    text = "memory routes concepts through compact cortex mesh traces. " * 4
    tokenizer = CharTokenizer.from_text(text)
    model = tiny_model(tokenizer)
    factory = TextCorpusFactory(text, tokenizer, seq_len=model.config.max_seq_len, seed=17)
    trainer = Trainer(
        model,
        tokenizer,
        lr=5e-3,
        seq_len=model.config.max_seq_len,
        factory=factory,
        eval_factory=TextCorpusFactory(text, tokenizer, seq_len=model.config.max_seq_len, seed=23),
    )

    report = trainer.train_steps(steps=16, batch_size=4, fixed_batch=True, eval_batches=1)

    assert report["skipped_steps"] == 0
    assert report["last_loss"] < report["first_loss"]
    assert report["eval_after"]["loss"] <= report["eval_before"]["loss"]
