"""Koan 08: supervised fine-tuning data and head tuning.

This koan starts the fine-tuning path with two concrete mechanics:

1. Format chat-style SFT examples so loss is computed only on assistant tokens.
2. Freeze a reusable base model and update only a small task head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import TODO

ChatMessage = dict[str, str]


class TinyBaseTextClassifier(nn.Module):
    """A tiny base model used by the fine-tuning koans.

    Treat `embedding` + `encoder` as the reusable base model, and `classifier` as
    the task head. It is intentionally small so tests can verify which parameters
    update after each fine-tuning method.
    """

    def __init__(self, vocab_size: int, d_model: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids: Tensor) -> Tensor:
        X = self.embedding(token_ids)
        hidden = torch.tanh(self.encoder(X))
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


class TinyCausalLM(nn.Module):
    """A toy causal LM for SFT, distillation, and DPO exercises."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.lm_head(torch.tanh(self.embedding(input_ids)))


def encode_chat_messages(
    messages: list[ChatMessage], vocab: dict[str, int], eos_token: str = "<eos>"
) -> Tensor:
    """Encode chat messages with explicit role tokens.

    Example format:
        <user> hello <eos> <assistant> hi <eos>

    This is intentionally simple, but it mirrors the real SFT requirement: apply
    the same chat template at training time that the model will see at inference.
    """
    ids = []
    for message in messages:
        ids.append(vocab[f"<{message['role']}>" ])
        ids.extend(vocab[token] for token in message["content"].split())
        ids.append(vocab[eos_token])
    return torch.tensor(ids, dtype=torch.long)


def assistant_only_labels(
    input_ids: Tensor,
    assistant_token_id: int,
    eos_token_id: int,
    ignore_index: int = -100,
) -> Tensor:
    """Create SFT labels that train only on assistant content and assistant EOS.

    Prompt/user tokens should be ignored. After an assistant role token appears,
    the following content tokens and the closing EOS are labels. The assistant
    role token itself is still ignored.
    """
    labels = torch.full_like(input_ids, ignore_index)
    assistant = False
    for i, token in enumerate(input_ids.tolist()):
        if token == assistant_token_id:
            assistant = True
        elif assistant:
            labels[i] = token
            if token == eos_token_id:
                assistant = False
    return labels


def sft_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Run one supervised fine-tuning step for next-token prediction.

    logits[:, :-1] predict labels[:, 1:]. Use ignore_index so prompt tokens do
    not contribute to the loss.
    """
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=ignore_index,
    )
    loss.backward()
    optimizer.step()
    return loss.detach()


def freeze_base_for_classifier_tuning(
    model: TinyBaseTextClassifier,
) -> TinyBaseTextClassifier:
    """Freeze the reusable base and leave only the classifier head trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model


def supervised_finetune_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: Tensor,
    labels: Tensor,
) -> Tensor:
    """Run one supervised classifier fine-tuning step with cross-entropy loss."""
    optimizer.zero_grad()
    loss = F.cross_entropy(model(token_ids), labels)
    loss.backward()
    optimizer.step()
    return loss.detach()
