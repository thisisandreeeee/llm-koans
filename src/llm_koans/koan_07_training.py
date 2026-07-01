"""Koan 07: assembling and training transformer variants.

Koans 00–06 built attention mechanics by hand — Q/K/V projections, multi‑head
reshaping, scaled dot‑product scores, causal masks, encoder/decoder blocks, and
layer‑norm residuals.  Those koans teach what happens inside the black box.

This koan shows how a practitioner actually wires up transformers in PyTorch:
use `nn` building blocks, compose them into real architectures, and run a
standard training loop.  The point is to move from “I can spell out attention”
to “I can build any transformer variant and train it.”

The three exercises progress naturally through the transformer taxonomy:

    Exercise A — TinyTransformer (GPT‑style causal LM)
    Exercise B — TinyEncoder      (BERT‑style bidirectional encoder)
    Exercise C — TinyEncoderDecoder (T5‑style seq‑to‑seq with cross‑attention)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import TODO

# ── shared helpers ────────────────────────────────────────────────────────────

def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Upper‑triangular mask: tokens cannot attend to future positions."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Exercise A — TinyTransformer (GPT‑style causal language model)
# ═══════════════════════════════════════════════════════════════════════════════


class TinyTransformer(nn.Module):
    """A minimal GPT‑style transformer for causal language modelling.

    Components (provided):
        self.token_embedding    – nn.Embedding(vocab_size, d_model)
        self.position_embedding – nn.Embedding(max_seq_len, d_model)
        self.transformer        – nn.TransformerEncoder (stack of TransformerEncoderLayer)
        self.lm_head            – nn.Linear(d_model, vocab_size)

    You implement `forward`.  The model should:
    1. embed tokens + positions
    2. build a causal (upper‑triangular) attention mask
    3. run the encoder with that mask
    4. project to vocabulary logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return next‑token logits of shape (batch, seq_len, vocab_size)."""
        _, T = token_ids.shape
        tokens = self.token_embedding(token_ids)  # (B, T, d_model)
        positions = self.position_embedding(torch.arange(T, device=token_ids.device))  # (T, d_model)
        X = tokens + positions
        logits = self.transformer(X, mask=_causal_mask(T, device=token_ids.device))
        return self.lm_head(logits)


# ═══════════════════════════════════════════════════════════════════════════════
# Exercise B — TinyEncoder (BERT‑style bidirectional encoder)
# ═══════════════════════════════════════════════════════════════════════════════


class TinyEncoder(nn.Module):
    """A BERT‑style bidirectional transformer for classification.

    Same building blocks as TinyTransformer, but the forward pass differs:
    - **no** causal mask → every token can attend to every other token
    - mean‑pool the token representations
    - feed pooled vector through a classification head

    Components (provided):
        self.token_embedding    – nn.Embedding(vocab_size, d_model)
        self.position_embedding – nn.Embedding(max_seq_len, d_model)
        self.transformer        – nn.TransformerEncoder (no mask = bidirectional)
        self.classifier         – nn.Linear(d_model, num_classes)

    You implement `forward`.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        max_seq_len: int,
        num_classes: int,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return class logits of shape (batch, num_classes)."""
        _, T = token_ids.shape
        tokens = self.token_embedding(token_ids)
        positions = self.position_embedding(torch.arange(T, device=token_ids.device))
        logits = self.transformer(tokens + positions).mean(dim=1)
        return self.classifier(logits)


# ═══════════════════════════════════════════════════════════════════════════════
# Exercise C — TinyEncoderDecoder (T5‑style seq‑to‑seq with cross‑attention)
# ═══════════════════════════════════════════════════════════════════════════════


class TinyEncoderDecoder(nn.Module):
    """A T5‑style encoder‑decoder transformer for sequence‑to‑sequence tasks.

    The encoder processes the source sequence bidirectionally.
    The decoder generates the target sequence auto‑regressively, attending both
    to its own previous tokens (causal self‑attention) and to the full encoder
    output (cross‑attention).

    Components (provided):
        Encoder side:
            self.enc_token_embedding    – nn.Embedding(src_vocab_size, d_model)
            self.enc_position_embedding – nn.Embedding(max_seq_len, d_model)
            self.encoder                – nn.TransformerEncoder (bidirectional)
        Decoder side:
            self.dec_token_embedding    – nn.Embedding(tgt_vocab_size, d_model)
            self.dec_position_embedding – nn.Embedding(max_seq_len, d_model)
            self.decoder                – nn.TransformerDecoder
            self.lm_head                – nn.Linear(d_model, tgt_vocab_size)

    You implement `forward`.  The model should:
    1. encode source → memory (no causal mask)
    2. decode target → causally masked self‑attention over target + cross‑attention to memory
    3. project decoder output to target vocabulary logits
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        # --- encoder ---
        self.enc_token_embedding = nn.Embedding(src_vocab_size, d_model)
        self.enc_position_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- decoder ---
        self.dec_token_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.dec_position_embedding = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src_ids: Tensor, tgt_ids: Tensor) -> Tensor:
        """Return next‑token logits of shape (batch, tgt_seq_len, tgt_vocab_size)."""
        _, T_src = src_ids.shape
        src_tokens = self.enc_token_embedding(src_ids)
        src_positions = self.enc_position_embedding(torch.arange(T_src, device=src_ids.device))
        memory = self.encoder(src_tokens + src_positions)

        _, T_tgt = tgt_ids.shape
        tgt_tokens = self.dec_token_embedding(tgt_ids)
        tgt_positions = self.dec_position_embedding(torch.arange(T_tgt, device=tgt_ids.device))
        decoder_hidden = self.decoder(
            tgt_tokens + tgt_positions, memory, tgt_mask=_causal_mask(T_tgt, device=tgt_ids.device)
        )
        return self.lm_head(decoder_hidden)


# ═══════════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════════


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: Tensor,
) -> Tensor:
    """Run one next‑token prediction training step.

    logits[:, :-1, :] should predict token_ids[:, 1:].  Use cross‑entropy loss,
    back‑propagate, and step the optimizer.
    """
    optimizer.zero_grad()
    logits = model(token_ids)  # (B, T, D)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits[:, :-1, :].transpose(1, 2), token_ids[:, 1:])
    loss.backward()
    optimizer.step()
    return loss.detach()


def parameter_delta_norm(before: dict[str, Tensor], after: dict[str, Tensor]) -> float:
    """Return the total L2 norm of parameter changes between two state‑dict snapshots."""
    total = 0.0

    for key in before:
        diff = after[key] - before[key]
        total += (diff**2).sum().item()

    return total**0.5
