"""Experimental standalone memory components.

This module is intentionally not imported by the main CortexMesh model. The
classes here are small PyTorch-compatible experiments that can be tested in
isolation before any future integration decision.
"""

from __future__ import annotations

import torch
from torch import nn


class HolographicMemory(nn.Module):
    """Holographic reduced-representation memory with circular binding.

    Keys and values are associated by circular convolution. A memory trace is
    the sum of bound key/value pairs, and approximate recall is performed by
    circular correlation with a query key.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be at least 2")
        self.dim = dim
        self.eps = eps

    def _check_last_dim(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.shape[-1] != self.dim:
            raise ValueError(f"{name} last dimension must be {self.dim}")

    def _unitary_key(self, key: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(key, n=self.dim, dim=-1)
        magnitude = spectrum.abs().clamp_min(self.eps)
        return torch.fft.irfft(spectrum / magnitude, n=self.dim, dim=-1)

    def bind(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Bind key and value tensors with matching shape and last dimension."""

        self._check_last_dim(key, "key")
        self._check_last_dim(value, "value")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")

        unit_key = self._unitary_key(key)
        key_spectrum = torch.fft.rfft(unit_key, n=self.dim, dim=-1)
        value_spectrum = torch.fft.rfft(value, n=self.dim, dim=-1)
        return torch.fft.irfft(key_spectrum * value_spectrum, n=self.dim, dim=-1)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Approximately recover a value from a bound trace and its key."""

        self._check_last_dim(bound, "bound")
        self._check_last_dim(key, "key")

        unit_key = self._unitary_key(key)
        bound_spectrum = torch.fft.rfft(bound, n=self.dim, dim=-1)
        key_spectrum = torch.fft.rfft(unit_key, n=self.dim, dim=-1)
        return torch.fft.irfft(bound_spectrum * key_spectrum.conj(), n=self.dim, dim=-1)

    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Store key/value pairs and recall values for query keys.

        Args:
            keys: Tensor shaped ``[batch, items, dim]``.
            values: Tensor shaped ``[batch, items, dim]``.
            queries: Tensor shaped ``[batch, queries, dim]``.

        Returns:
            Tensor shaped ``[batch, queries, dim]`` with approximate recalls.
        """

        if keys.ndim != 3:
            raise ValueError("keys must have shape [batch, items, dim]")
        if values.shape != keys.shape:
            raise ValueError("values must have the same shape as keys")
        if queries.ndim != 3:
            raise ValueError("queries must have shape [batch, queries, dim]")
        if queries.shape[0] != keys.shape[0]:
            raise ValueError("queries batch size must match keys batch size")
        self._check_last_dim(keys, "keys")
        self._check_last_dim(queries, "queries")

        trace = self.bind(keys, values).sum(dim=1)
        return self.unbind(trace.unsqueeze(1), queries)


HolographicBinder = HolographicMemory
