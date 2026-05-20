from __future__ import annotations

import torch
import torch.nn.functional as F

from cortexmesh.experimental import HolographicBinder, HolographicMemory


def test_holographic_bind_unbind_shapes() -> None:
    memory = HolographicMemory(dim=32)
    key = torch.randn(2, 5, 32)
    value = torch.randn(2, 5, 32)

    bound = memory.bind(key, value)
    recalled = memory.unbind(bound, key)

    assert bound.shape == (2, 5, 32)
    assert recalled.shape == (2, 5, 32)


def test_holographic_forward_shapes() -> None:
    memory = HolographicBinder(dim=64)
    keys = torch.randn(3, 4, 64)
    values = torch.randn(3, 4, 64)
    queries = torch.randn(3, 2, 64)

    recalled = memory(keys, values, queries)

    assert recalled.shape == (3, 2, 64)


def test_holographic_single_pair_recall_is_close() -> None:
    torch.manual_seed(11)
    memory = HolographicMemory(dim=128)
    key = torch.randn(4, 128)
    value = torch.randn(4, 128)

    recalled = memory.unbind(memory.bind(key, value), key)
    similarity = F.cosine_similarity(recalled, value, dim=-1)

    assert torch.all(similarity > 0.98)


def test_holographic_memory_recalls_matching_values_approximately() -> None:
    torch.manual_seed(17)
    memory = HolographicMemory(dim=512)
    keys = torch.randn(2, 2, 512)
    values = F.normalize(torch.randn(2, 2, 512), dim=-1)

    recalled = memory(keys, values, keys)
    pairwise_similarity = torch.einsum("bqd,bvd->bqv", F.normalize(recalled, dim=-1), values)
    target_similarity = pairwise_similarity.diagonal(dim1=1, dim2=2)
    non_target_similarity = pairwise_similarity.masked_fill(
        torch.eye(2, dtype=torch.bool).unsqueeze(0),
        -1.0,
    )

    assert torch.all(target_similarity > 0.65)
    assert torch.all(target_similarity > non_target_similarity.amax(dim=-1))
