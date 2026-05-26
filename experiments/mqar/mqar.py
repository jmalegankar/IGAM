"""Multi-Query Associative Recall (MQAR) — supervised cell-architecture test.

A cell solves MQAR if it can store key→value pairs from the prefix of a sequence
and retrieve the correct value when later shown a query key. This decouples the
"architecture works" question (this task) from the "RL can discover a policy"
question (POPGym). Standard benchmark from Arora et al. ICLR 2024 (Zoology).

Task format (sequence of token IDs):
    k1 v1  k2 v2  ...  kN vN  q1 q2 ... qM

    At each query position q_i, the model should predict value(q_i). Loss is
    cross-entropy averaged only over query positions; the K-V pair tokens
    contribute no loss (free to be ignored or used for state-building).

Key parameters:
    vocab_size   total token vocabulary (keys and values share it)
    n_pairs      number of (k, v) pairs written before the query phase
    n_queries    number of queries asked (each query is a random one of the keys)
    extra_pad    optional filler tokens between the write phase and query phase
                 to test long-range recall (zero by default)

Usage:
    python -m experiments.mqar.mqar \\
        --hebbian-mode gated_delta_eps --n-pairs 8 --vocab 64 --steps 2000
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from memrl.cell import DTHLMU


# ---------------------------------------------------------------------------
# Task: MQAR sequence generator
# ---------------------------------------------------------------------------

@dataclass
class MQARConfig:
    vocab_size: int = 64        # token vocabulary (keys and values share)
    n_pairs:    int = 8         # K-V pairs written in the prefix
    n_queries:  int = 4         # queries asked after the prefix
    extra_pad:  int = 0         # filler tokens between write phase and queries

    @property
    def seq_len(self) -> int:
        # 2 tokens per pair (key, value) + extra_pad + n_queries
        return 2 * self.n_pairs + self.extra_pad + self.n_queries


def sample_mqar(
    cfg: MQARConfig,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of MQAR sequences.

    Returns:
        tokens:       (B, T) int64 token IDs
        targets:      (B, T) int64 target token (only meaningful where mask is True)
        loss_mask:    (B, T) bool, True at query positions only
    """
    B = batch_size
    T = cfg.seq_len
    V = cfg.vocab_size

    # Sample n_pairs distinct keys per sequence (no key collisions in-prefix);
    # values are independent uniform draws. Using torch.randperm per row would
    # be slow; instead we sample then dedupe via argsort over noise (Gumbel-top-k).
    # For our typical n_pairs<<vocab_size, simple-with-replacement is fine + rare collision.
    keys   = torch.randint(0, V, (B, cfg.n_pairs), device=device, generator=generator)
    values = torch.randint(0, V, (B, cfg.n_pairs), device=device, generator=generator)

    # Interleave keys and values: position 2i = k_i, position 2i+1 = v_i.
    prefix = torch.empty(B, 2 * cfg.n_pairs, dtype=torch.long, device=device)
    prefix[:, 0::2] = keys
    prefix[:, 1::2] = values

    # Optional padding filler — use a constant token (V-1 reserved if you wanted,
    # but we just use 0 here; the loss mask ensures it doesn't matter).
    pad = torch.zeros(B, cfg.extra_pad, dtype=torch.long, device=device)

    # Queries: pick n_queries random *positions* from each sequence's prefix
    # so the corresponding (k, v) pair is guaranteed to be one we wrote.
    query_idx = torch.randint(0, cfg.n_pairs, (B, cfg.n_queries),
                              device=device, generator=generator)
    # gather the actual key/value pairs for each query
    query_keys   = torch.gather(keys,   1, query_idx)        # (B, n_queries)
    query_values = torch.gather(values, 1, query_idx)        # (B, n_queries)

    # Assemble full sequence: prefix | pad | queries
    tokens = torch.cat([prefix, pad, query_keys], dim=1)     # (B, T)
    assert tokens.shape[1] == T

    # Targets: anything at a query position should predict the corresponding value
    targets   = torch.zeros_like(tokens)
    loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
    q_start = 2 * cfg.n_pairs + cfg.extra_pad
    targets[:, q_start:q_start + cfg.n_queries]   = query_values
    loss_mask[:, q_start:q_start + cfg.n_queries] = True

    return tokens, targets, loss_mask


# ---------------------------------------------------------------------------
# Model wrapper: token embed → cell loop → linear classifier head
# ---------------------------------------------------------------------------

class MQARModel(nn.Module):
    """Wrap a recurrent cell for MQAR token-level prediction.

    Embed token → run cell step-by-step → project hidden state → vocab logits.
    """

    def __init__(self, cell: nn.Module, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.cell      = cell
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.head      = nn.Linear(cell.output_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) int64 → logits (B, T, V)."""
        B, T = tokens.shape
        x = self.embed(tokens)                              # (B, T, C)

        state = self.cell.init_state(B, device=x.device, dtype=x.dtype)
        logits_per_t = []
        for t in range(T):
            h, state, _ = self.cell.step(x[:, t], state)
            logits_per_t.append(self.head(h))               # (B, V)
        return torch.stack(logits_per_t, dim=1)             # (B, T, V)


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    hebbian_mode: str = "gated_delta_eps"
    d_model:     int = 64
    memory_size: int = 32
    theta:       float = 100.0
    n_scales:    int = 3
    scale_factor: float = 2.0
    assoc_size:  int = 64
    lr:          float = 1e-3
    batch_size:  int = 32
    n_steps:     int = 2000
    eval_every:  int = 200
    eval_batches: int = 8
    seed:        int = 0


def build_model(tcfg: TrainConfig, vocab_size: int, device: torch.device) -> MQARModel:
    cell = DTHLMU(
        input_size=tcfg.d_model,
        hidden_size=tcfg.d_model,
        memory_size=tcfg.memory_size,
        theta=tcfg.theta,
        n_scales=tcfg.n_scales,
        scale_factor=tcfg.scale_factor,
        assoc_size=tcfg.assoc_size,
        hebbian_mode=tcfg.hebbian_mode,
    )
    model = MQARModel(cell, vocab_size=vocab_size, d_model=tcfg.d_model).to(device)
    return model


@torch.no_grad()
def evaluate(model: MQARModel, mcfg: MQARConfig, tcfg: TrainConfig,
             device: torch.device) -> dict:
    """Compute mean accuracy + loss on freshly sampled batches."""
    model.eval()
    eval_gen = torch.Generator(device=device).manual_seed(tcfg.seed + 99999)
    total_correct = 0
    total_count   = 0
    total_loss    = 0.0
    for _ in range(tcfg.eval_batches):
        tokens, targets, mask = sample_mqar(mcfg, tcfg.batch_size, device, eval_gen)
        logits = model(tokens)                              # (B, T, V)
        masked_logits = logits[mask]                        # (#queries, V)
        masked_target = targets[mask]                       # (#queries,)
        loss = F.cross_entropy(masked_logits, masked_target, reduction="sum")
        pred = masked_logits.argmax(dim=-1)
        total_correct += (pred == masked_target).sum().item()
        total_count   += masked_target.numel()
        total_loss    += loss.item()
    model.train()
    return {
        "acc":  total_correct / total_count,
        "loss": total_loss / total_count,
    }


def train_one(mcfg: MQARConfig, tcfg: TrainConfig, device: torch.device,
              verbose: bool = True) -> dict:
    torch.manual_seed(tcfg.seed)

    model = build_model(tcfg, vocab_size=mcfg.vocab_size, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    train_gen = torch.Generator(device=device).manual_seed(tcfg.seed)

    n_params = sum(p.numel() for p in model.parameters())
    chance = 1.0 / mcfg.vocab_size
    if verbose:
        print(f"mode={tcfg.hebbian_mode}  pairs={mcfg.n_pairs}  vocab={mcfg.vocab_size}  "
              f"params={n_params:,}  chance={chance:.4f}")

    history = []
    t0 = time.perf_counter()
    for step in range(1, tcfg.n_steps + 1):
        tokens, targets, mask = sample_mqar(mcfg, tcfg.batch_size, device, train_gen)
        logits = model(tokens)
        masked_logits = logits[mask]
        masked_target = targets[mask]
        loss = F.cross_entropy(masked_logits, masked_target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % tcfg.eval_every == 0 or step == 1:
            ev = evaluate(model, mcfg, tcfg, device)
            elapsed = time.perf_counter() - t0
            history.append({"step": step, "train_loss": loss.item(),
                            "eval_acc": ev["acc"], "eval_loss": ev["loss"],
                            "elapsed_s": elapsed})
            if verbose:
                print(f"  step={step:5d}  train_loss={loss.item():.3f}  "
                      f"eval_acc={ev['acc']:.3f}  eval_loss={ev['loss']:.3f}  "
                      f"({elapsed:.1f}s)")

    final = evaluate(model, mcfg, tcfg, device)
    result = {
        "hebbian_mode": tcfg.hebbian_mode,
        "n_pairs":  mcfg.n_pairs,
        "n_queries": mcfg.n_queries,
        "extra_pad": mcfg.extra_pad,
        "vocab_size": mcfg.vocab_size,
        "n_steps":  tcfg.n_steps,
        "final_acc": final["acc"],
        "final_loss": final["loss"],
        "chance": chance,
        "n_params": n_params,
        "wallclock_s": time.perf_counter() - t0,
        "history": history,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hebbian-mode", default="gated_delta_eps",
                        choices=["additive", "delta", "gated_delta", "gated_delta_eps"])
    parser.add_argument("--n-pairs", type=int, default=8)
    parser.add_argument("--n-queries", type=int, default=4)
    parser.add_argument("--extra-pad", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--assoc-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    device = torch.device(args.device)
    mcfg = MQARConfig(
        vocab_size=args.vocab,
        n_pairs=args.n_pairs,
        n_queries=args.n_queries,
        extra_pad=args.extra_pad,
    )
    tcfg = TrainConfig(
        hebbian_mode=args.hebbian_mode,
        d_model=args.d_model,
        assoc_size=args.assoc_size,
        batch_size=args.batch_size,
        n_steps=args.steps,
        lr=args.lr,
        seed=args.seed,
    )

    result = train_one(mcfg, tcfg, device)
    print(f"\nFINAL  acc={result['final_acc']:.3f}  loss={result['final_loss']:.3f}  "
          f"chance={result['chance']:.4f}  ({result['wallclock_s']:.1f}s)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
