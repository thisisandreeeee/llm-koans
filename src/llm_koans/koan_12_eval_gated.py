"""Koan 12: evaluation-gated fine-tuning."""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from .common import TODO


def classification_accuracy(
    model: nn.Module, token_ids: Tensor, labels: Tensor
) -> float:
    """Evaluate a classifier on a tiny validation batch."""
    TODO(
        "Run model under no_grad, argmax logits, and return mean accuracy as a Python float."
    )


def accept_candidate_if_improves(
    model: nn.Module,
    candidate_state: dict[str, Tensor],
    val_token_ids: Tensor,
    val_labels: Tensor,
    min_accuracy: float,
    min_delta: float = 0.0,
) -> bool:
    """Gate a fine-tuned candidate on validation quality.

    Save the current model state, evaluate the baseline, load the candidate, and
    keep it only if candidate accuracy is at least min_accuracy and improves over
    the baseline by min_delta. Otherwise restore the original state.
    """
    TODO(
        "Evaluate baseline and candidate states, keep candidate only if it clears both gates; otherwise restore baseline and return False."
    )
