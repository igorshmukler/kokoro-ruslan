"""
Checkpoint regression analysis script.
For each checkpoint, extracts:
  - epoch, loss, optimizer_steps_completed, ema_updates
  - weight norm / mean / std / max-abs for every named parameter
  - weight change (L2 norm delta) from the previous checkpoint
  - any NaN or Inf weights
Prints a sorted summary table and flags the epoch with the largest
weight jump (likely where regression started).
"""

import sys
import os
# Make kokoro package importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
import re
import math
from pathlib import Path


CHECKPOINT_DIR = Path("my_model")
# Key layers to report individually (substring match against param name)
KEY_LAYERS = [
    "mel_projection",
    "mel_linear",
    "duration_predictor.linear",
    "pitch_predictor.linear",
    "energy_predictor.linear",
    "decoder.layers.0.self_attn.w_o",
    "decoder.layers.5.self_attn.w_o",
    "text_embedding",
]


def param_stats(tensor):
    t = tensor.float()
    has_nan = bool(torch.isnan(t).any())
    has_inf = bool(torch.isinf(t).any())
    return {
        "norm": t.norm(2).item(),
        "mean": t.mean().item(),
        "std": t.std().item(),
        "max_abs": t.abs().max().item(),
        "nan": has_nan,
        "inf": has_inf,
    }


def load_checkpoints():
    files = sorted(
        CHECKPOINT_DIR.glob("checkpoint_epoch_*.pth"),
        key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)),
    )
    records = []
    for f in files:
        epoch_num = int(re.search(r"epoch_(\d+)", f.name).group(1))
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[LOAD ERROR] {f.name}: {e}")
            continue

        rec = {
            "file": f.name,
            "epoch_num": epoch_num,
            "ck_epoch": ck.get("epoch", "?"),
            "loss": ck.get("loss", float("nan")),
            "optimizer_steps": ck.get("optimizer_steps_completed", ck.get("current_optimizer_step", "?")),
            "ema_updates": ck.get("ema_updates", "?"),
            "state_dict": ck.get("model_state_dict") or ck.get("state_dict") or {},
            "ema_state_dict": ck.get("ema_model_state_dict") or {},
        }
        records.append(rec)
    return records


def compute_weight_stats(records):
    """Add per-parameter stats and delta from previous checkpoint."""
    prev_state = None
    for rec in records:
        sd = rec["state_dict"]
        stats = {}
        any_nan = False
        any_inf = False
        total_norm = 0.0
        delta_norm = 0.0
        n_params = 0

        for name, tensor in sd.items():
            s = param_stats(tensor)
            stats[name] = s
            if s["nan"]:
                any_nan = True
            if s["inf"]:
                any_inf = True
            total_norm += s["norm"] ** 2
            n_params += tensor.numel()

            # Delta from previous
            if prev_state is not None and name in prev_state:
                diff = (tensor.float() - prev_state[name].float()).norm(2).item()
                stats[name]["delta"] = diff
                delta_norm += diff ** 2
            else:
                stats[name]["delta"] = None

        rec["param_stats"] = stats
        rec["total_weight_norm"] = math.sqrt(total_norm)
        rec["total_delta_norm"] = math.sqrt(delta_norm) if prev_state else None
        rec["any_nan"] = any_nan
        rec["any_inf"] = any_inf
        rec["n_params"] = n_params
        prev_state = {k: v.clone() for k, v in sd.items()}
    return records


def print_summary_table(records):
    print("\n" + "=" * 110)
    print(f"{'File':<38} {'Ep':>3} {'Loss':>10} {'Steps':>7} {'EMA Upd':>8} {'Wt Norm':>10} {'Delta Norm':>11} {'NaN':>4} {'Inf':>4}")
    print("=" * 110)
    for r in records:
        loss_str = f"{r['loss']:.6f}" if isinstance(r['loss'], float) and not math.isnan(r['loss']) else str(r['loss'])
        delta_str = f"{r['total_delta_norm']:.4f}" if r['total_delta_norm'] is not None else "  --  "
        nan_flag = "YES" if r["any_nan"] else "."
        inf_flag = "YES" if r["any_inf"] else "."
        print(f"{r['file']:<38} {r['epoch_num']:>3} {loss_str:>10} {str(r['optimizer_steps']):>7} "
              f"{str(r['ema_updates']):>8} {r['total_weight_norm']:>10.2f} {delta_str:>11} {nan_flag:>4} {inf_flag:>4}")
    print("=" * 110)


def print_key_layer_table(records):
    print("\n--- Key layer weight norms per epoch ---")
    # Header
    epoch_labels = [f"Ep{r['epoch_num']:02d}" for r in records]
    print(f"{'Layer':<60} " + "  ".join(f"{e:>8}" for e in epoch_labels))
    print("-" * (60 + 10 * len(records)))

    # Collect all layer names that match key layers
    all_names = list(records[0]["param_stats"].keys()) if records else []
    matched = [n for n in all_names if any(k in n for k in KEY_LAYERS)]

    for name in matched:
        norms = []
        for r in records:
            s = r["param_stats"].get(name)
            norms.append(f"{s['norm']:>8.3f}" if s else "       ?")
        print(f"{name:<60} " + "  ".join(norms))


def print_key_layer_delta_table(records):
    print("\n--- Key layer weight DELTAS (L2 change from previous checkpoint) ---")
    epoch_labels = [f"Ep{r['epoch_num']:02d}" for r in records]
    print(f"{'Layer':<60} " + "  ".join(f"{e:>8}" for e in epoch_labels))
    print("-" * (60 + 10 * len(records)))

    all_names = list(records[0]["param_stats"].keys()) if records else []
    matched = [n for n in all_names if any(k in n for k in KEY_LAYERS)]

    for name in matched:
        deltas = []
        for r in records:
            s = r["param_stats"].get(name)
            if s and s.get("delta") is not None:
                deltas.append(f"{s['delta']:>8.4f}")
            else:
                deltas.append("    --  ")
        print(f"{name:<60} " + "  ".join(deltas))


def print_biggest_deltas(records):
    """For each epoch transition, print the top-10 parameters with largest weight change."""
    print("\n--- Top-10 largest weight changes at each epoch transition ---")
    for r in records:
        if r["total_delta_norm"] is None:
            continue
        deltas = [(name, s["delta"]) for name, s in r["param_stats"].items() if s.get("delta") is not None]
        deltas.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  → {r['file']}  (total delta norm: {r['total_delta_norm']:.4f})")
        for name, d in deltas[:10]:
            print(f"    {d:>10.5f}  {name}")


def flag_regression_epoch(records):
    """Find epoch with largest delta norm jump — likely where regression started."""
    candidates = [(r["total_delta_norm"], r) for r in records if r["total_delta_norm"] is not None]
    if not candidates:
        return
    candidates.sort(reverse=True)
    top = candidates[:3]
    print("\n--- Largest overall weight changes (top 3 candidates for regression onset) ---")
    for delta, r in top:
        print(f"  Epoch {r['epoch_num']:>3}  ({r['file']})  delta norm = {delta:.4f}  loss = {r['loss']}")


def main():
    if not CHECKPOINT_DIR.is_dir():
        print(f"Directory not found: {CHECKPOINT_DIR}")
        return

    print(f"Loading checkpoints from {CHECKPOINT_DIR} ...")
    records = load_checkpoints()
    if not records:
        print("No checkpoints found.")
        return

    print(f"Found {len(records)} checkpoint(s). Computing weight statistics...")
    records = compute_weight_stats(records)

    print_summary_table(records)
    print_key_layer_table(records)
    print_key_layer_delta_table(records)
    print_biggest_deltas(records)
    flag_regression_epoch(records)

    # NaN/Inf report
    bad = [r for r in records if r["any_nan"] or r["any_inf"]]
    if bad:
        print("\n!!! CHECKPOINTS WITH NaN or Inf WEIGHTS !!!")
        for r in bad:
            print(f"  {r['file']}  nan={r['any_nan']}  inf={r['any_inf']}")
    else:
        print("\nNo NaN or Inf weights found in any checkpoint. Good.")


if __name__ == "__main__":
    main()
