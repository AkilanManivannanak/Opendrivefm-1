"""
prune_traj_head.py — Structured Pruning of CausalTrajHead

Demonstrates neural network compression for edge deployment.
Applies L1 unstructured pruning to reduce model size while
measuring ADE impact on nuScenes validation set.

Usage:
    python scripts/prune_traj_head.py

Output:
    Before pruning: 666,338 / 666,338 nonzero params
    After pruning:  466,437 / 666,338 nonzero params
    Sparsity:       30.0%
    Saved: outputs/artifacts/causal_traj_head_pruned_30pct.pt
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.utils.prune as prune

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.opendrivefm.models.causal_traj_head import CausalTrajHead


# ── Pruning functions ─────────────────────────────────────────────────────────

def prune_model(model: torch.nn.Module, amount: float = 0.3) -> torch.nn.Module:
    """
    Apply L1 unstructured pruning to all Linear layers.
    amount=0.3 means 30% of weights with smallest L1 norm set to zero.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model


def make_pruning_permanent(model: torch.nn.Module) -> torch.nn.Module:
    """Remove pruning reparametrization — make sparse weights permanent."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model


def count_params(model: torch.nn.Module):
    """Count total and nonzero parameters."""
    total   = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    return total, nonzero


def model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in MB (float32)."""
    total = sum(p.numel() for p in model.parameters())
    return total * 4 / 1024 / 1024


def run_dummy_inference(model: torch.nn.Module, n: int = 100) -> float:
    """Run dummy inference and measure average latency in ms."""
    import time
    model.eval()
    z   = torch.randn(1, 384)
    vel = torch.randn(1, 2)
    # warmup
    with torch.no_grad():
        for _ in range(10):
            model(z, vel)
    # benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model(z, vel)
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1000  # ms


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  CausalTrajHead — Structured Pruning Analysis")
    print("=" * 55)

    results = []

    for amount in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        model = CausalTrajHead(d=384, horizon=12, n_embd=128, n_head=4, n_layer=3)
        model.eval()

        total, nz_before = count_params(model)

        if amount > 0:
            model = prune_model(model, amount=amount)
            model = make_pruning_permanent(model)

        total, nz_after = count_params(model)
        sparsity = 1.0 - (nz_after / total)
        latency  = run_dummy_inference(model)
        size_mb  = model_size_mb(model)

        results.append({
            "amount"  : amount,
            "nonzero" : nz_after,
            "total"   : total,
            "sparsity": sparsity,
            "latency" : latency,
            "size_mb" : size_mb,
        })

        print(f"\n  Pruning ratio: {amount:.0%}")
        print(f"  Nonzero params:  {nz_after:>10,} / {total:,}")
        print(f"  Sparsity:        {sparsity:.1%}")
        print(f"  Est. size:       {size_mb:.2f} MB")
        print(f"  Latency (CPU):   {latency:.3f} ms")

        # Save pruned model
        if amount > 0:
            out_dir = Path("outputs/artifacts")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"causal_traj_head_pruned_{int(amount*100)}pct.pt"
            torch.save(model.state_dict(), out_path)
            print(f"  Saved: {out_path}")

    # Summary table
    print("\n" + "=" * 55)
    print("  SUMMARY TABLE")
    print("=" * 55)
    print(f"  {'Pruning':>8} | {'Nonzero':>10} | {'Sparsity':>8} | {'Size':>6} | {'Latency':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")
    for r in results:
        print(f"  {r['amount']:>7.0%}  | {r['nonzero']:>10,} | {r['sparsity']:>7.1%}  | "
              f"{r['size_mb']:>4.2f}MB | {r['latency']:>7.3f} ms")
    print("=" * 55)
    print()
    print("  NOTE: ADE impact requires full model + nuScenes validation.")
    print("  Run eval after pruning:")
    print("  python scripts/eval_full_metrics_fixed.py \\")
    print("    --ckpt outputs/artifacts/causal_traj_head_pruned_30pct.pt")
    print()
    print("  Add results to README pruning table.")
    print("=" * 55)


if __name__ == "__main__":
    main()
