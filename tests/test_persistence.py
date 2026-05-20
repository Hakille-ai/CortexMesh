from __future__ import annotations

import json

import torch

from cortexmesh import CortexMesh, CortexMeshConfig


def tiny_config() -> CortexMeshConfig:
    return CortexMeshConfig(
        vocab_size=12,
        signal_dim=8,
        concept_dim=8,
        memory_slots=4,
        cycles=1,
        route_hidden_dim=12,
        max_seq_len=8,
    )


def test_config_dict_round_trip() -> None:
    config = tiny_config()

    restored = CortexMeshConfig.from_dict(config.to_dict())

    assert restored == config


def test_config_json_round_trip(tmp_path) -> None:
    config = tiny_config()
    path = tmp_path / "config.json"

    config.save_json(path)
    restored = CortexMeshConfig.load_json(path)

    assert json.loads(path.read_text(encoding="utf-8")) == config.to_dict()
    assert restored == config


def test_model_save_and_load_pretrained_cpu(tmp_path) -> None:
    torch.manual_seed(7)
    model = CortexMesh(tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 5))

    model.save_pretrained(tmp_path)
    restored = CortexMesh.from_pretrained(tmp_path)

    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "pytorch_model.bin").exists()
    assert restored.config == model.config
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)

    model.eval()
    restored.eval()
    with torch.no_grad():
        output = model(tokens, return_internal=False)
        restored_output = restored(tokens, return_internal=False)

    assert output.keys() == restored_output.keys()
    for key, value in output.items():
        assert torch.allclose(restored_output[key], value)
