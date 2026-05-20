"""Training loop for CortexMesh synthetic tasks."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any

import torch
import torch.nn.functional as F

from .data import CharTokenizer, SyntheticTaskFactory
from .evaluation import (
    compute_recall_accuracy,
    compute_rule_accuracy,
    compute_token_accuracy,
    cycle_delta_norm,
    memory_slot_entropy,
)
from .model import CortexMesh


@dataclass
class LossBreakdown:
    total: float
    token: float
    reconstruction: float
    rule: float
    recall: float

    @classmethod
    def mean(cls, breakdowns: list["LossBreakdown"]) -> "LossBreakdown":
        if not breakdowns:
            raise ValueError("breakdowns must not be empty")
        scale = 1.0 / len(breakdowns)
        return cls(
            total=sum(item.total for item in breakdowns) * scale,
            token=sum(item.token for item in breakdowns) * scale,
            reconstruction=sum(item.reconstruction for item in breakdowns) * scale,
            rule=sum(item.rule for item in breakdowns) * scale,
            recall=sum(item.recall for item in breakdowns) * scale,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "total": self.total,
            "token": self.token,
            "reconstruction": self.reconstruction,
            "rule": self.rule,
            "recall": self.recall,
        }


class Trainer:
    """Small CPU-friendly trainer for the bundled synthetic tasks."""

    def __init__(
        self,
        model: CortexMesh,
        tokenizer: CharTokenizer | None = None,
        lr: float = 3e-3,
        seq_len: int | None = None,
        device: str | torch.device | None = None,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.02,
        factory: Any | None = None,
        eval_factory: Any | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or CharTokenizer()
        self.device = torch.device(device or "cpu")
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing
        self.model.to(self.device)
        task_seq_len = seq_len or model.config.max_seq_len
        self.factory = factory or SyntheticTaskFactory(self.tokenizer, seq_len=task_seq_len)
        self.eval_factory = eval_factory or SyntheticTaskFactory(
            self.tokenizer,
            seq_len=task_seq_len,
            seed=101,
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, LossBreakdown]:
        output = self.model(batch["inputs"])
        return self._compute_loss_from_output(output, batch)

    def _compute_loss_from_output(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, LossBreakdown]:
        vocab_size = output["logits"].shape[-1]
        token_loss = F.cross_entropy(
            output["logits"].reshape(-1, vocab_size),
            batch["targets"].reshape(-1),
            label_smoothing=self.label_smoothing,
        )
        reconstruction_loss = F.mse_loss(output["reconstruction"], output["signals"].detach())

        rule_mask = batch["rule_mask"]
        if rule_mask.any():
            rule_loss = F.cross_entropy(
                output["rule_logits"][rule_mask],
                batch["rule_labels"][rule_mask],
                label_smoothing=self.label_smoothing,
            )
        else:
            rule_loss = token_loss.new_zeros(())

        recall_mask = batch["recall_mask"]
        if recall_mask.any():
            recall_loss = F.cross_entropy(
                output["recall_logits"][recall_mask],
                batch["recall_targets"][recall_mask],
                label_smoothing=self.label_smoothing,
            )
        else:
            recall_loss = token_loss.new_zeros(())

        total = token_loss + 0.10 * reconstruction_loss + 0.20 * rule_loss + 0.20 * recall_loss
        return total, LossBreakdown(
            total=float(total.detach().cpu()),
            token=float(token_loss.detach().cpu()),
            reconstruction=float(reconstruction_loss.detach().cpu()),
            rule=float(rule_loss.detach().cpu()),
            recall=float(recall_loss.detach().cpu()),
        )

    def evaluate(
        self,
        batches: int = 4,
        batch_size: int = 16,
        include_metrics: bool = False,
    ) -> dict[str, object]:
        """Run a small deterministic evaluation pass without updating weights."""

        if batches < 1:
            raise ValueError("batches must be at least 1")
        was_training = self.model.training
        eval_rng_state = _get_rng_state(self.eval_factory)
        self.model.eval()
        breakdowns: list[LossBreakdown] = []
        metric_batches: list[dict[str, float]] = []
        with torch.no_grad():
            for _ in range(batches):
                batch = _make_batch(self.eval_factory, batch_size, self.device, fixed_cycle=True).as_dict()
                output = self.model(batch["inputs"])
                _, breakdown = self._compute_loss_from_output(output, batch)
                breakdowns.append(breakdown)
                if include_metrics:
                    metric_batches.append(_output_metrics(output, batch))
        _restore_rng_state(self.eval_factory, eval_rng_state)
        if was_training:
            self.model.train()

        mean = LossBreakdown.mean(breakdowns)
        result: dict[str, object] = {
            "loss": mean.total,
            "breakdown": mean,
            "breakdown_dict": mean.as_dict(),
            "batches": batches,
            "batch_size": batch_size,
        }
        if include_metrics:
            result["metrics"] = _mean_metrics(metric_batches)
        return result

    def train_steps(
        self,
        steps: int = 30,
        batch_size: int = 16,
        fixed_batch: bool = False,
        eval_batches: int = 4,
    ) -> dict[str, object]:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if eval_batches < 0:
            raise ValueError("eval_batches must be non-negative")

        eval_before = self.evaluate(eval_batches, batch_size) if eval_batches else None
        self.model.train()
        fixed = None
        if fixed_batch:
            fixed = self.factory.make_batch(batch_size, self.device, fixed_cycle=True).as_dict()

        losses: list[float] = []
        skipped_steps = 0
        last_breakdown: LossBreakdown | None = None
        for step in range(steps):
            batch = fixed or _make_batch(
                self.factory,
                batch_size,
                self.device,
                fixed_cycle=True,
                step=step,
            ).as_dict()
            self.optimizer.zero_grad(set_to_none=True)
            loss, breakdown = self.compute_loss(batch)
            if not torch.isfinite(loss):
                skipped_steps += 1
                losses.append(losses[-1] if losses else float("nan"))
                continue
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            losses.append(breakdown.total)
            last_breakdown = breakdown

        eval_after = self.evaluate(eval_batches, batch_size) if eval_batches else None
        smoothed_losses = _moving_average(losses)

        return {
            "losses": losses,
            "smoothed_losses": smoothed_losses,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "last_breakdown": last_breakdown,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "eval_delta": (
                eval_after["loss"] - eval_before["loss"]
                if eval_before is not None and eval_after is not None
                else None
            ),
            "skipped_steps": skipped_steps,
        }


def _moving_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    smoothed: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        span = values[start : idx + 1]
        smoothed.append(sum(span) / len(span))
    return smoothed


def _output_metrics(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        "token_accuracy": compute_token_accuracy(output, batch),
        "rule_accuracy": compute_rule_accuracy(output, batch),
        "recall_accuracy": compute_recall_accuracy(output, batch),
        "memory_slot_entropy": memory_slot_entropy(output),
        "cycle_delta_norm": cycle_delta_norm(output),
    }


def _mean_metrics(metric_batches: list[dict[str, float]]) -> dict[str, float]:
    if not metric_batches:
        return {}
    mean: dict[str, float] = {}
    for key in metric_batches[0]:
        values = [metrics[key] for metrics in metric_batches if not math.isnan(metrics[key])]
        mean[key] = sum(values) / len(values) if values else float("nan")
    return mean


def _make_batch(
    factory: Any,
    batch_size: int,
    device: torch.device,
    fixed_cycle: bool,
    step: int | None = None,
) -> Any:
    signature = inspect.signature(factory.make_batch)
    if "step" in signature.parameters:
        return factory.make_batch(batch_size, device, fixed_cycle=fixed_cycle, step=step)
    return factory.make_batch(batch_size, device, fixed_cycle=fixed_cycle)


def _get_rng_state(factory: Any) -> object | None:
    rng = _factory_rng(factory)
    return rng.getstate() if rng is not None else None


def _restore_rng_state(factory: Any, state: object | None) -> None:
    rng = _factory_rng(factory)
    if rng is not None and state is not None:
        rng.setstate(state)


def _factory_rng(factory: Any) -> Any | None:
    if hasattr(factory, "rng"):
        return factory.rng
    synthetic = getattr(factory, "synthetic_factory", None)
    if synthetic is not None and hasattr(synthetic, "rng"):
        return synthetic.rng
    return None
