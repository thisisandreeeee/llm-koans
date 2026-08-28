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
    token_logps = F.log_softmax(logits[:, :-1], dim=-1).gather(
        -1, target_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    if mask is not None:
        token_logps = token_logps.masked_fill(~mask[:, 1:], 0)
    return token_logps.sum(dim=-1)


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
    policy_gap = policy_chosen_logp - policy_rejected_logp
    reference_gap = reference_chosen_logp - reference_rejected_logp
    return -F.logsigmoid(beta * (policy_gap - reference_gap)).mean()


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
    optimizer.zero_grad()
    policy_chosen = sequence_logprobs(policy(prompt_chosen_ids), prompt_chosen_ids, chosen_mask)
    policy_rejected = sequence_logprobs(
        policy(prompt_rejected_ids), prompt_rejected_ids, rejected_mask
    )
    with torch.no_grad():
        reference_chosen = sequence_logprobs(
            reference(prompt_chosen_ids), prompt_chosen_ids, chosen_mask
        )
        reference_rejected = sequence_logprobs(
            reference(prompt_rejected_ids), prompt_rejected_ids, rejected_mask
        )
    loss = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta,
    )
    loss.backward()
    optimizer.step()
    return loss.detach()
