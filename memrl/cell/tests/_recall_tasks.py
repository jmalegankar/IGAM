"""Synthetic associative-recall task: MQAR + training harness.

MQAR (Multi-Query Associative Recall) is the canonical sanity benchmark
for associative-memory architectures. Input is a sequence of (key, value)
pairs followed by query keys; the model must predict the matching value
for each query.

Reference: Arora et al. 2024, "Zoology: Measuring and Improving Recall in
Efficient Language Models" (https://arxiv.org/abs/2312.04927) — used MQAR
to demonstrate that linear-attention cells differ markedly in associative
recall capacity even when they match each other on perplexity.

For Phase A, MQAR validates that the cell can do basic key-value lookup
under PPO-scale models (tiny hidden_size, short sequences). The harder
tests come at the Phase A benchmark sweep on POPGym.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from memrl.cell.base import RecurrentCell


# Sentinel for "ignore" positions in cross-entropy loss.
# Standard torch convention: targets at IGNORE_INDEX positions are not
# included in the loss or the accuracy metric.
IGNORE_INDEX = -100


def make_mqar_batch(
    batch_size: int,
    n_pairs: int,
    n_queries: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Tensor, Tensor]:
    """Generate one MQAR batch.

    Layout (T = 2 · n_pairs + n_queries):
        [k_1, v_1, k_2, v_2, ..., k_N, v_N, q_1, q_2, ..., q_M]

    Targets:
        - At query positions: the matching value for that key.
        - At all other positions: IGNORE_INDEX (not scored).

    Keys are sampled WITHOUT replacement so each key appears at most once
    in the (k, v) pairs. Queries are drawn from those keys so every query
    has a definitive answer. Values may repeat.

    Returns:
        inputs:  (B, T)  long tensor of token ids
        targets: (B, T)  long tensor; IGNORE_INDEX at non-query positions
    """
    T = 2 * n_pairs + n_queries
    inputs = torch.zeros(batch_size, T, dtype=torch.long, device=device)
    targets = torch.full(
        (batch_size, T), IGNORE_INDEX, dtype=torch.long, device=device
    )

    for b in range(batch_size):
        # Unique keys for this sequence.
        keys = torch.randperm(vocab_size, device=device)[:n_pairs]
        # Values may repeat — that's fine, the model just learns the
        # *most recent* k→v mapping for each k.
        values = torch.randint(0, vocab_size, (n_pairs,), device=device)

        # Place (k, v) pairs in alternating positions.
        inputs[b, 0 : 2 * n_pairs : 2] = keys
        inputs[b, 1 : 2 * n_pairs : 2] = values

        # Queries sampled from the placed keys.
        query_idx = torch.randint(0, n_pairs, (n_queries,), device=device)
        for j in range(n_queries):
            pos = 2 * n_pairs + j
            inputs[b, pos] = keys[query_idx[j]]
            targets[b, pos] = values[query_idx[j]]

    return inputs, targets


class TokenRecurrentModel(nn.Module):
    """Wrap a RecurrentCell with an embedding and a classifier head.

    Used to make cells composable with the synthetic recall tasks: tokens
    in, logits out. The cell itself doesn't know about tokens — it sees
    (T, B, input_size) continuous embeddings.
    """

    def __init__(self, cell: RecurrentCell, vocab_size: int) -> None:
        super().__init__()
        self.cell = cell
        self.embed = nn.Embedding(vocab_size, cell.input_size)
        # Classifier shares no weight with the embedding — standard
        # token-prediction setup, kept untied for simplicity.
        self.classifier = nn.Linear(cell.output_size, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, T) long → logits: (B, T, vocab_size)"""
        B, T = tokens.shape
        x_seq = self.embed(tokens).transpose(0, 1)  # (T, B, input_size)
        s = self.cell.init_state(B, device=tokens.device)
        y_seq, _, _ = self.cell.forward_sequence(x_seq, s)
        return self.classifier(y_seq.transpose(0, 1))  # (B, T, vocab_size)


def train_recall(
    cell: RecurrentCell,
    *,
    vocab_size: int,
    n_pairs: int,
    n_queries: int,
    batch_size: int,
    n_steps: int,
    lr: float = 3e-3,
    grad_clip: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Train `cell` on MQAR and return (initial_loss, final_loss, final_accuracy).

    `final_accuracy` is computed on the QUERY positions of the final batch
    (chance = 1/vocab_size). Gradient clipping is applied per the README's
    discipline for matrix-memory cells under PPO-style training.
    """
    model = TokenRecurrentModel(cell, vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    initial_loss: float | None = None
    final_loss: float = float("nan")
    final_accuracy: float = 0.0

    for step in range(n_steps):
        x, y = make_mqar_batch(
            batch_size=batch_size,
            n_pairs=n_pairs,
            n_queries=n_queries,
            vocab_size=vocab_size,
            device=device,
        )

        logits = model(x)  # (B, T, V)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        if step == 0:
            initial_loss = loss.item()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()
        final_loss = loss.item()

    # Final accuracy: argmax predictions at query positions of the last batch.
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = y != IGNORE_INDEX
        if mask.any():
            final_accuracy = (preds[mask] == y[mask]).float().mean().item()

    return {
        "initial_loss": float(initial_loss) if initial_loss is not None else float("nan"),
        "final_loss": float(final_loss),
        "final_accuracy": float(final_accuracy),
    }
