from __future__ import annotations

import json
import subprocess
import sys

from cortexmesh.train_text import main


def test_train_text_module_json_smoke(tmp_path) -> None:
    save_dir = tmp_path / "model"
    command = [
        sys.executable,
        "-m",
        "cortexmesh.train_text",
        "--text",
        "cortex mesh custom text cli smoke test. " * 2,
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--seq-len",
        "8",
        "--seed",
        "31",
        "--save-dir",
        str(save_dir),
        "--json-output",
    ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)

    assert report["source"] == "inline"
    assert report["steps"] == 1
    assert report["batch_size"] == 2
    assert report["seq_len"] == 8
    assert report["skipped_steps"] == 0
    assert len(report["sample"]) > 0
    assert (save_dir / "config.json").exists()
    assert (save_dir / "pytorch_model.bin").exists()
    assert (save_dir / "tokenizer.json").exists()


def test_train_text_console_entrypoint_accepts_text_file(tmp_path, capsys) -> None:
    text_path = tmp_path / "corpus.txt"
    text_path.write_text("text file corpus for cortex mesh cli. " * 2, encoding="utf-8")

    main(
        [
            "--text-file",
            str(text_path),
            "--steps",
            "1",
            "--batch-size",
            "2",
            "--seq-len",
            "8",
            "--json-output",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert report["source"] == str(text_path)
    assert report["source_chars"] == len(text_path.read_text(encoding="utf-8"))
