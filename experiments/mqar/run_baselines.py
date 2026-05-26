"""Baseline comparison: published-style DeltaNet vs our DTH-LMU (full and Hebbian-only).

Hypothesis under test:
  - The DTH-LMU's MQAR ceiling (~28% on easy) is caused by:
    (a) reading with q = W_Q(h_prev) instead of q = W_q(x_t),  AND/OR
    (b) the four-branch output dilutes the Hebbian signal.

If PureDeltaNet hits ~95% on the same easy task:
    → task wiring is fine; the architecture *can* solve it.
If Hebbian-only DTH-LMU substantially closes the gap toward DeltaNet:
    → (b) was the problem; restructure the output routing.
If Hebbian-only still tops out at ~30%:
    → (a) is the problem; refactor read head to query from x_t.

Each baseline is trained on the EASY MQAR config (4 pairs, no gap, vocab=32)
for 5000 steps. Single seed. Total ~15-20 min on CPU.

Usage:
    python -m experiments.mqar.run_baselines
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.mqar.baselines import PureDeltaNetCell, hebbian_only_dthlmu
from experiments.mqar.mqar import (
    MQARConfig, MQARModel, TrainConfig, evaluate, sample_mqar,
)


RESULTS_DIR = Path(__file__).parent / "_results"


def train_with_cell(cell, mcfg: MQARConfig, tcfg: TrainConfig,
                    device: torch.device, label: str) -> dict:
    """Train an arbitrary cell on MQAR; return final accuracy + trajectory."""
    torch.manual_seed(tcfg.seed)
    model = MQARModel(cell, vocab_size=mcfg.vocab_size, d_model=tcfg.d_model).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tcfg.lr)
    gen = torch.Generator(device=device).manual_seed(tcfg.seed)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    chance = 1.0 / mcfg.vocab_size
    print(f"[{label}] params={n_params:,} trainable={n_trainable:,} chance={chance:.4f}")

    history = []
    t0 = time.perf_counter()
    for step in range(1, tcfg.n_steps + 1):
        tokens, targets, mask = sample_mqar(mcfg, tcfg.batch_size, device, gen)
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits[mask], targets[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % tcfg.eval_every == 0 or step == 1:
            ev = evaluate(model, mcfg, tcfg, device)
            elapsed = time.perf_counter() - t0
            history.append({"step": step, "eval_acc": ev["acc"],
                            "eval_loss": ev["loss"], "elapsed_s": elapsed})
            print(f"  [{label}] step={step:5d} train_loss={loss.item():.3f} "
                  f"acc={ev['acc']:.3f} ({elapsed:.1f}s)")

    final = evaluate(model, mcfg, tcfg, device)
    return {
        "label": label,
        "final_acc": final["acc"],
        "final_loss": final["loss"],
        "chance": chance,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "wallclock_s": time.perf_counter() - t0,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--vocab", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    mcfg = MQARConfig(vocab_size=args.vocab, n_pairs=4, n_queries=4, extra_pad=0)
    tcfg = TrainConfig(
        d_model=64, assoc_size=64,
        batch_size=args.batch_size,
        n_steps=args.steps, seed=args.seed,
    )

    print(f"MQAR EASY: {mcfg}")
    print(f"steps={args.steps}, vocab={args.vocab}, device={args.device}\n")

    results = []
    t0 = time.perf_counter()

    # ── Baseline 1: published-style pure DeltaNet ────────────────────────────
    print("\n=== [1/3] PureDeltaNet (q from x_t, single head) ===")
    cell = PureDeltaNetCell(input_size=tcfg.d_model, hidden_size=tcfg.d_model,
                             assoc_size=tcfg.assoc_size)
    r = train_with_cell(cell, mcfg, tcfg, device, label="pure_deltanet")
    results.append(r)
    print(f"  → final acc={r['final_acc']:.3f}\n")

    # ── Baseline 2: Hebbian-only DTH-LMU (additive mode) ─────────────────────
    print("\n=== [2/3] DTH-LMU(additive) with aux branches frozen at 0 ===")
    cell = hebbian_only_dthlmu(
        input_size=tcfg.d_model, hidden_size=tcfg.d_model,
        hebbian_mode="additive",
        memory_size=tcfg.memory_size, theta=tcfg.theta,
        n_scales=tcfg.n_scales, scale_factor=tcfg.scale_factor,
        assoc_size=tcfg.assoc_size,
    )
    r = train_with_cell(cell, mcfg, tcfg, device, label="dthlmu_additive_hebonly")
    results.append(r)
    print(f"  → final acc={r['final_acc']:.3f}\n")

    # ── Baseline 3: Hebbian-only DTH-LMU (gated_delta_eps mode) ──────────────
    print("\n=== [3/3] DTH-LMU(gated_delta_eps) with aux branches frozen at 0 ===")
    cell = hebbian_only_dthlmu(
        input_size=tcfg.d_model, hidden_size=tcfg.d_model,
        hebbian_mode="gated_delta_eps",
        memory_size=tcfg.memory_size, theta=tcfg.theta,
        n_scales=tcfg.n_scales, scale_factor=tcfg.scale_factor,
        assoc_size=tcfg.assoc_size,
    )
    r = train_with_cell(cell, mcfg, tcfg, device, label="dthlmu_gated_delta_eps_hebonly")
    results.append(r)
    print(f"  → final acc={r['final_acc']:.3f}\n")

    elapsed = time.perf_counter() - t0
    print("=" * 72)
    print(f"DONE in {elapsed/60:.1f} min")
    print("=" * 72)
    print(f"{'baseline':<40}  {'acc':>8}  {'vs chance':>10}  {'trainable':>10}")
    print("-" * 72)
    chance = 1.0 / args.vocab
    for r in results:
        ratio = r["final_acc"] / chance
        print(f"{r['label']:<40}  {r['final_acc']:>8.3f}  {ratio:>9.1f}×  {r['n_trainable']:>10,}")
    print(f"\n(chance = {chance:.4f})")

    # Verdict
    deltanet_acc = next(r["final_acc"] for r in results if r["label"] == "pure_deltanet")
    hebonly_accs = [r["final_acc"] for r in results if "hebonly" in r["label"]]
    print("\n=== diagnostic ===")
    if deltanet_acc < 0.7:
        print(f"⚠ PureDeltaNet only hit {deltanet_acc:.2f}. Task wiring might be wrong.")
    else:
        print(f"✓ PureDeltaNet hit {deltanet_acc:.2f} — task is solvable.")
        best_hebonly = max(hebonly_accs)
        if best_hebonly > 0.7:
            print(f"✓ Hebbian-only DTH-LMU also high ({best_hebonly:.2f}) — the cell's "
                  "Hebbian path works once aux branches are removed.")
            print("  → fix: restructure output to not dilute Hebbian signal.")
        elif best_hebonly > 0.4:
            print(f"~ Hebbian-only intermediate ({best_hebonly:.2f}) — aux branches "
                  "hurt but the read head also needs work.")
            print("  → fix: route query through x_t AND restructure output.")
        else:
            print(f"✗ Hebbian-only still stuck at {best_hebonly:.2f}.")
            print("  → fix: read head queries h_{t-1}; needs to query x_t. Architectural refactor.")

    out_path = Path(args.out) if args.out else (
        RESULTS_DIR / f"baselines_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "elapsed_s": elapsed,
        "task": {"n_pairs": mcfg.n_pairs, "n_queries": mcfg.n_queries,
                 "extra_pad": mcfg.extra_pad, "vocab_size": mcfg.vocab_size},
        "n_steps": args.steps,
        "results": results,
    }, indent=2))
    print(f"\nresults: {out_path}")


if __name__ == "__main__":
    main()
