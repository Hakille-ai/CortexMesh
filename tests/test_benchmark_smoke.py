from __future__ import annotations

import json

from benchmarks.run_benchmark import main, run_benchmark


def test_tiny_benchmark_smoke_runs_quickly() -> None:
    result = run_benchmark(config_name="tiny", steps=1, batch_size=2, seed=123, eval_batches=1)

    assert result["config"] == "tiny"
    assert result["device"] == "cpu"
    assert result["steps"] == 1
    assert result["batch_size"] == 2
    assert result["train_seconds"] >= 0.0
    assert result["train_last_loss"] > 0.0
    assert result["eval_loss"] is not None
    assert result["eval_loss"] > 0.0
    assert result["skipped_steps"] == 0


def test_benchmark_cli_can_write_json(tmp_path) -> None:
    output = tmp_path / "benchmark.json"
    result = main(
        [
            "--config",
            "tiny",
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--seed",
            "123",
            "--json-output",
            str(output),
        ]
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["config"] == result["config"]
    assert written["train_last_loss"] == result["train_last_loss"]
    assert written["eval_loss"] == result["eval_loss"]
