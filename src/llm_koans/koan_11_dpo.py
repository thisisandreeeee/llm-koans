"""Koan 11: DPO preference fine-tuning."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import TODO


def sequence_logprobs(
    logits: Tensor, target_ids: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Return total log-probability of each target sequence.

    logits has shape (B, T, V), target_ids has shape (B, T). Each position's
    logits predict the following token. If mask is given, only target positions
    where mask is True contribute to the sequence score.
    """
    TODO(
        "Shift causal logits against the following target tokens, then sum only the optional mask's target positions."
    )


def dpo_loss(
    policy_chosen_logp: Tensor,
    policy_rejected_logp: Tensor,
    reference_chosen_logp: Tensor,
    reference_rejected_logp: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Direct Preference Optimization loss for preference fine-tuning.

    DPO increases the policy's chosen-vs-rejected log-probability gap relative to
    a frozen reference model, without training a separate reward model.
    """
    TODO(
        "Compare the policy's preference margin with the reference model's margin, then form the DPO objective."
    )


def dpo_step(
    policy: nn.Module,
    reference: nn.Module,
    optimizer: torch.optim.Optimizer,
    prompt_chosen_ids: Tensor,
    prompt_rejected_ids: Tensor,
    chosen_mask: Tensor,
    rejected_mask: Tensor,
    beta: float = 0.1,
) -> Tensor:
    """Run one DPO update on policy while keeping reference frozen.

    The tensors include prompt + completion. The masks should mark completion
    tokens only, so prompt likelihood does not become the preference signal.
    """
    TODO(
        "Evaluate both completions under policy and frozen reference, then optimize only the policy preference."
    )
