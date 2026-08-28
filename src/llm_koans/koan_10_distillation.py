"""Koan 10: distillation fine-tuning."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import TODO


def distillation_kl_loss(
    student_logits: Tensor, teacher_logits: Tensor, temperature: float = 1.0
) -> Tensor:
    """Compute softened teacher-to-student KL loss.

    The teacher provides a distribution over alternatives, not just a hard label.
    Temperature softens both distributions and the usual loss is scaled by T^2.
    """
    TODO(
        "Compare the student's and teacher's softened class distributions with the standard temperature scaling."
    )


def blended_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    hard_labels: Tensor,
    alpha: float,
    temperature: float = 1.0,
) -> Tensor:
    """Blend hard-label CE with soft teacher KL.

    alpha is the weight on distillation loss; 1-alpha is the weight on hard-label
    supervised loss.
    """
    TODO(
        "Combine hard-label supervision and teacher imitation according to alpha's documented meaning."
    )


def distillation_step(
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: Tensor,
    hard_labels: Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> Tensor:
    """Run one distillation update on the student while keeping teacher frozen."""
    TODO(
        "Update only the student from the blended objective; teacher inference must not build gradients."
    )
