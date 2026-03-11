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
import argparse
# Make kokoro package importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
import re
import math
import statistics
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


CHECKPOINT_DIR = Path("my_model")
TB_LOG_DIR = Path("my_model/logs")

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
    # Perform statistics under no_grad to avoid building/retaining autograd graph
    with torch.no_grad():
        has_nan = bool(torch.isnan(t).any())
        has_inf = bool(torch.isinf(t).any())
        # Compute statistics defensively: use population std (unbiased=False)
        # to avoid Bessel-correction warnings when tensor has <=1 element.
        numel = t.numel()
        if numel == 0:
            mean_val = 0.0
            std_val = 0.0
            max_abs = 0.0
            norm_val = 0.0
        else:
            mean_val = t.mean().item()
            std_val = t.std(unbiased=False).item()
            max_abs = t.abs().max().item()
            norm_val = t.norm(2).item()

    return {
        "norm": norm_val,
        "mean": mean_val,
        "std": std_val,
        "max_abs": max_abs,
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
            "config": ck.get("config"),
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

        # Compute stats and deltas without tracking gradients
        with torch.no_grad():
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


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard analysis
# ─────────────────────────────────────────────────────────────────────────────

def _load_tb(log_dir: Path):
    """Load EventAccumulator; return None if unavailable or dir missing."""
    if not _TB_AVAILABLE:
        print("\n[TensorBoard] tensorboard package not installed — skipping TB analysis.")
        return None
    if not log_dir.is_dir():
        print(f"\n[TensorBoard] Log directory not found: {log_dir} — skipping TB analysis.")
        return None
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def _get(ea, tag):
    """Return list of (step, value) for a scalar tag, or []."""
    try:
        return [(e.step, e.value) for e in ea.Scalars(tag)]
    except Exception:
        return []


def _series_stats(series):
    """Return dict of summary statistics for a (step, value) list."""
    if not series:
        return None
    vals = [v for _, v in series]
    return {
        "n": len(vals),
        "first": vals[0],
        "last": vals[-1],
        "mean": statistics.mean(vals),
        "min": min(vals),
        "max": max(vals),
        "step_first": series[0][0],
        "step_last": series[-1][0],
        "trend": "UP ▲" if vals[-1] > vals[0] else "DOWN ▼",
        "delta": vals[-1] - vals[0],
    }


def _pct(series, p):
    """Return the p-th percentile (0..100) of values in series."""
    vals = sorted(v for _, v in series)
    idx = max(0, min(len(vals) - 1, int(len(vals) * p / 100)))
    return vals[idx]


def tb_print_step_loss_summary(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Step-level Training Loss Summary")
    print("=" * 90)
    tags = ["loss/total", "loss/mel", "loss/stop", "loss/duration", "loss/pitch", "loss/energy"]
    fmt = f"  {'Tag':<28} {'N':>5}  {'Steps':>10}  {'First':>9}  {'Last':>9}  {'Δ':>9}  {'Trend':>7}  {'Mean':>9}  {'Min':>9}  {'Max':>9}"
    print(fmt)
    print("  " + "-" * 106)
    for tag in tags:
        s = _series_stats(_get(ea, tag))
        if s is None:
            print(f"  {tag:<28}  (no data)")
            continue
        print(f"  {tag:<28} {s['n']:>5}  "
              f"{s['step_first']:>5}–{s['step_last']:<5}  "
              f"{s['first']:>9.5f}  {s['last']:>9.5f}  "
              f"{s['delta']:>+9.5f}  {s['trend']:>7}  "
              f"{s['mean']:>9.5f}  {s['min']:>9.5f}  {s['max']:>9.5f}")


def tb_print_epoch_table(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Epoch-level Train vs Val Metrics")
    print("=" * 90)
    epoch_tags = [
        ("mel",      "loss/train_mel_epoch",      "loss/val_mel_epoch"),
        ("total",    "loss/train_total_epoch",    "loss/val_total_epoch"),
        ("stop",     "loss/train_stop_epoch",     "loss/val_stop_epoch"),
        ("duration", "loss/train_duration_epoch", "loss/val_duration_epoch"),
        ("spec_conv","metrics/train_spectral_convergence", "metrics/val_spectral_convergence"),
        ("f0_rmse",  None,                        "metrics/val_f0_rmse"),
    ]
    # Collect all epoch steps from val_mel
    val_mel_series = _get(ea, "loss/val_mel_epoch")
    if not val_mel_series:
        print("  No epoch-level data available yet.")
        return
    epoch_steps = [s for s, _ in val_mel_series]
    n_epochs = len(epoch_steps)

    print(f"  {'Metric':<12}  " + "  ".join(f"Ep? step={s:>5}" for s in epoch_steps))
    # Replace 'Ep?' with actual epoch index
    header = f"  {'Metric':<12}  " + "  ".join(f"  Ep{i+1:02d} (s{s:04d})" for i, s in enumerate(epoch_steps))
    print("  " + "-" * (14 + 18 * n_epochs))

    for label, train_tag, val_tag in epoch_tags:
        train_vals = {s: v for s, v in (_get(ea, train_tag) if train_tag else [])}
        val_vals   = {s: v for s, v in _get(ea, val_tag)} if val_tag else {}

        if train_tag:
            row_t = f"  {label+' train':<12}  "
            for s in epoch_steps:
                v = train_vals.get(s)
                row_t += f"  {'?':>12}  " if v is None else f"  {v:>12.5f}  "
            print(row_t)

        if val_tag:
            row_v = f"  {label+' val':<12}  "
            vals_for_step = [val_vals.get(s) for s in epoch_steps]
            for i, (s, v) in enumerate(zip(epoch_steps, vals_for_step)):
                flag = ""
                if i > 0:
                    prev = vals_for_step[i - 1]
                    if v is not None and prev is not None and v > prev:
                        flag = " ▲"  # regression indicator
                row_v += f"  {'?':>12}  " if v is None else f"  {v:>10.5f}{flag:2s}  "
            print(row_v)
        print()


def tb_print_stop_token_analysis(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Stop Token Analysis")
    print("=" * 90)
    series = _get(ea, "loss/stop")
    if not series:
        print("  No step-level stop data.")
    else:
        vals = [v for _, v in series]
        p50 = _pct(series, 50)
        p90 = _pct(series, 90)
        p99 = _pct(series, 99)
        burst_thresh = p50 * 2
        bursts = [(s, v) for s, v in series if v > burst_thresh]
        # find when bursts stop
        late_bursts = [(s, v) for s, v in bursts if s > (series[-1][0] * 0.5)]
        print(f"  Step-level loss/stop ({len(series)} points, steps {series[0][0]}–{series[-1][0]})")
        print(f"    first={vals[0]:.5f}  last={vals[-1]:.5f}  Δ={vals[-1]-vals[0]:+.5f}  "
              f"mean={statistics.mean(vals):.5f}")
        print(f"    min={min(vals):.5f}  max={max(vals):.5f}")
        print(f"    p50={p50:.5f}  p90={p90:.5f}  p99={p99:.5f}")
        print(f"    bursts > 2×median ({burst_thresh:.4f}):  total={len(bursts)}  "
              f"in 2nd half={len(late_bursts)}")
        if bursts:
            print(f"    burst steps: {[s for s, _ in bursts[:15]]}" +
                  (" ..." if len(bursts) > 15 else ""))
        if late_bursts:
            print(f"    ⚠  Late bursts (2nd half of run): {late_bursts[:8]}")
        else:
            print(f"    ✓  No bursts in 2nd half of run — stop loss stabilizing.")

    # epoch-level stop
    for tag, label in [("loss/train_stop_epoch", "train"), ("loss/val_stop_epoch", "val")]:
        ep = _get(ea, tag)
        if ep:
            print(f"\n  Epoch-level stop ({label}):")
            for i, (s, v) in enumerate(ep):
                prev_v = ep[i-1][1] if i > 0 else None
                flag = " ▲ REGRESSION" if prev_v is not None and v > prev_v else ""
                print(f"    Ep{i+1:02d}  step={s:>5}  {v:.5f}{flag}")


def tb_print_gradient_analysis(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Gradient Health")
    print("=" * 90)
    gn   = _get(ea, "stats/grad_norm")
    gnc  = _get(ea, "stats/grad_norm_clipped")

    if not gn:
        print("  No grad_norm data.")
        return

    raw_vals    = [v for _, v in gn]
    clipped_vals= [v for _, v in gnc] if gnc else []

    print(f"  grad_norm  ({len(gn)} steps, {gn[0][0]}–{gn[-1][0]})")
    print(f"    first={raw_vals[0]:.4f}  last={raw_vals[-1]:.4f}  "
          f"mean={statistics.mean(raw_vals):.4f}  max={max(raw_vals):.4f}")

    for thresh in [5.0, 10.0, 20.0]:
        spikes = [(s, v) for s, v in gn if v > thresh]
        # split: early (first 10% of steps) vs late
        step_range = gn[-1][0] - gn[0][0]
        early_cutoff = gn[0][0] + step_range * 0.10
        late_spikes  = [(s, v) for s, v in spikes if s > early_cutoff]
        print(f"    spikes > {thresh:4.1f}: total={len(spikes):4d}  "
              f"after first 10% of steps={len(late_spikes):4d}" +
              (f"  e.g. {late_spikes[:4]}" if late_spikes else ""))

    if clipped_vals:
        max_clip = max(clipped_vals)
        at_cap   = sum(1 for v in clipped_vals if abs(v - max_clip) < 1e-4)
        pct      = 100.0 * at_cap / len(clipped_vals)
        status   = "✓ HEALTHY" if pct < 20 else ("⚠ ELEVATED" if pct < 40 else "✗ SATURATED")
        print(f"\n  Clipping saturation: {at_cap}/{len(clipped_vals)} steps at cap "
              f"({pct:.1f}%)  cap={max_clip:.4f}  → {status}")

        # Per-epoch saturation breakdown
        # Infer epoch boundaries from val_mel steps
        boundaries = [s for s, _ in _get(ea, "loss/val_mel_epoch")]
        if boundaries:
            print("  Per-epoch clipping saturation:")
            prev_b = 0
            for i, b in enumerate(boundaries):
                window = [(s, v) for s, v in gnc if prev_b < s <= b]
                if window:
                    w_cap = max(v for _, v in window)
                    w_sat = sum(1 for _, v in window if abs(v - w_cap) < 1e-4)
                    w_pct = 100.0 * w_sat / len(window)
                    flag  = "" if w_pct < 20 else (" ⚠" if w_pct < 40 else " ✗")
                    print(f"    Ep{i+1:02d} (steps {prev_b+1}–{b:>5}): "
                          f"{w_sat:>4}/{len(window):<5} = {w_pct:5.1f}%{flag}")
                prev_b = b
            # current partial epoch
            if gnc:
                rest = [(s, v) for s, v in gnc if s > prev_b]
                if rest:
                    r_cap = max(v for _, v in rest)
                    r_sat = sum(1 for _, v in rest if abs(v - r_cap) < 1e-4)
                    r_pct = 100.0 * r_sat / len(rest)
                    flag  = "" if r_pct < 20 else (" ⚠" if r_pct < 40 else " ✗")
                    print(f"    Ep?  (steps {prev_b+1}–{rest[-1][0]:>5}, in progress): "
                          f"{r_sat:>4}/{len(rest):<5} = {r_pct:5.1f}%{flag}")


def tb_print_lr_trajectory(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Learning Rate Trajectory")
    print("=" * 90)
    for tag, label in [("stats/lr_decoder", "decoder"), ("stats/lr_encoder", "encoder")]:
        series = _get(ea, tag)
        if not series:
            continue
        vals = [v for _, v in series]
        print(f"  {label}: first={vals[0]:.7f}  last={vals[-1]:.7f}  "
              f"max={max(vals):.7f}  trend={'UP ▲' if vals[-1] > vals[0] else 'DOWN ▼'}")
        # Print ~8 evenly-spaced samples
        n = len(series)
        step_size = max(1, n // 8)
        samples = series[::step_size] + [series[-1]]
        seen = set()
        for s, v in samples:
            if s not in seen:
                print(f"    step={s:>5}  {v:.8f}")
                seen.add(s)
    # Warmup diagnosis: check if LR is still monotonically rising over last 10 points
    dec = _get(ea, "stats/lr_decoder")
    if dec:
        max_lr  = max(v for _, v in dec)
        curr_lr = dec[-1][1]
        # Is LR still rising? Check if the last several points are increasing
        tail = [v for _, v in dec[-10:]]
        still_rising = len(tail) >= 2 and tail[-1] > tail[0]
        if still_rising and curr_lr >= max_lr * 0.99:
            print(f"\n  Phase: WARMUP (still rising, not yet at true peak ~{max_lr * 1.0 / (curr_lr / max_lr):.7f})")
        elif curr_lr < max_lr * 0.99:
            phase = "decay"
            pct   = 100.0 * curr_lr / max_lr
            print(f"\n  Phase: DECAY  (at {pct:.1f}% of peak LR {max_lr:.7f})")
        else:
            print(f"\n  Phase: AT PEAK  (lr={curr_lr:.7f})")


def tb_print_regression_flags(ea):
    """High-level PASS/FAIL flags for the most important regression signals."""
    print("\n" + "=" * 90)
    print("TENSORBOARD — Regression Flag Summary")
    print("=" * 90)
    flags = []

    # 1. Val mel trend across epochs
    vm = _get(ea, "loss/val_mel_epoch")
    if len(vm) >= 2:
        vals = [v for _, v in vm]
        if vals[-1] > vals[0]:
            flags.append(("FAIL", f"val_mel is HIGHER at last epoch ({vals[-1]:.5f}) vs first ({vals[0]:.5f})"))
        else:
            flags.append(("PASS", f"val_mel decreasing:  {vals[0]:.5f} → {vals[-1]:.5f}"))
        # Any epoch-to-epoch increase?
        regressions = [(i+2, vals[i+1] - vals[i]) for i in range(len(vals)-1) if vals[i+1] > vals[i]]
        if regressions:
            flags.append(("WARN", f"val_mel increased at epochs: " +
                          ", ".join(f"Ep{ep}+Δ{d:+.5f}" for ep, d in regressions)))

    # 2. Val/train mel gap (overfitting check)
    tm = _get(ea, "loss/train_mel_epoch")
    if vm and tm:
        gaps = []
        val_dict   = {s: v for s, v in vm}
        train_dict = {s: v for s, v in tm}
        for s in sorted(set(val_dict) & set(train_dict)):
            g = val_dict[s] - train_dict[s]
            gaps.append(g)
        if gaps:
            last_gap = gaps[-1]
            if last_gap > 0.5:
                flags.append(("WARN", f"Val/train mel gap = {last_gap:.4f} at last epoch (mild overfitting)"))
            elif last_gap > 1.0:
                flags.append(("FAIL", f"Val/train mel gap = {last_gap:.4f} — overfitting"))
            else:
                flags.append(("PASS", f"Val/train mel gap = {last_gap:.4f}"))

    # 3. Clipping saturation
    gnc = _get(ea, "stats/grad_norm_clipped")
    if gnc:
        max_clip = max(v for _, v in gnc)
        at_cap   = sum(1 for _, v in gnc if abs(v - max_clip) < 1e-4)
        pct      = 100.0 * at_cap / len(gnc)
        if pct > 40:
            flags.append(("FAIL", f"Clipping saturation {pct:.1f}% — model cannot take full gradient steps"))
        elif pct > 20:
            flags.append(("WARN", f"Clipping saturation {pct:.1f}% — elevated"))
        else:
            flags.append(("PASS", f"Clipping saturation {pct:.1f}%"))

    # 4. Late grad spikes (after first 10% of steps)
    gn = _get(ea, "stats/grad_norm")
    if gn:
        step_range   = gn[-1][0] - gn[0][0]
        early_cutoff = gn[0][0] + step_range * 0.10
        late_spikes  = [v for s, v in gn if s > early_cutoff and v > 5.0]
        if len(late_spikes) > 10:
            flags.append(("FAIL", f"{len(late_spikes)} grad spikes >5.0 after init warmup — outlier batches or loss imbalance"))
        elif len(late_spikes) > 3:
            flags.append(("WARN", f"{len(late_spikes)} late grad spikes >5.0"))
        else:
            flags.append(("PASS", f"Late grad spikes >5.0: {len(late_spikes)}"))

    # 5. Stop token balance vs mel
    stop_ep  = _get(ea, "loss/val_stop_epoch")
    mel_ep   = _get(ea, "loss/val_mel_epoch")
    if stop_ep and mel_ep:
        stop_dict = {s: v for s, v in stop_ep}
        mel_dict  = {s: v for s, v in mel_ep}
        common    = sorted(set(stop_dict) & set(mel_dict))
        if common:
            last_s = common[-1]
            ratio  = stop_dict[last_s] / mel_dict[last_s]
            if ratio > 0.5:
                flags.append(("WARN", f"val_stop/val_mel ratio = {ratio:.3f} — stop still dominates"))
            else:
                flags.append(("PASS", f"val_stop/val_mel ratio = {ratio:.3f}"))

    # 6. LR ramp coupling — was regression coincident with LR peak?
    if vm and len(vm) >= 3:
        dec_lr = _get(ea, "stats/lr_decoder")
        if dec_lr:
            max_lr     = max(v for _, v in dec_lr)
            peak_step  = next((s for s, v in dec_lr if abs(v - max_lr) < max_lr * 0.01), None)
            vm_vals    = [(s, v) for s, v in vm]
            if peak_step:
                regressions_post_peak = [(s, v) for s, v in vm_vals if s > peak_step]
                if len(regressions_post_peak) >= 2:
                    trend = regressions_post_peak[-1][1] - regressions_post_peak[0][1]
                    if trend > 0.05:
                        flags.append(("WARN", f"val_mel UP post LR peak (step {peak_step}): "
                                       f"Δ={trend:+.5f}"))
                    else:
                        flags.append(("PASS", f"val_mel stable/down post LR peak (step {peak_step})"))

    max_lbl = max(len(lbl) for lbl, _ in flags) if flags else 4
    for lbl, msg in flags:
        icon = "✓" if lbl == "PASS" else ("⚠" if lbl == "WARN" else "✗")
        print(f"  [{lbl:4s}] {icon}  {msg}")


def _collect_diagnostics(ea):
    """
    Gather all raw signals into a dict of structured findings.
    Used by both the flag summary and the recommendations section.
    """
    # Cache diagnostics on the EventAccumulator instance to avoid repeated TB reads
    try:
        cache_key = '_diag_cache'
        if ea is not None and hasattr(ea, cache_key):
            return getattr(ea, cache_key)
    except Exception:
        pass

    d = {}
    gn  = _get(ea, "stats/grad_norm")
    gnc = _get(ea, "stats/grad_norm_clipped")
    vm  = _get(ea, "loss/val_mel_epoch")
    tm  = _get(ea, "loss/train_mel_epoch")
    dec = _get(ea, "stats/lr_decoder")
    stop_step = _get(ea, "loss/stop")
    stop_ep   = _get(ea, "loss/val_stop_epoch")

    # ── grad spikes ─────────────────────────────────────────────────────────
    if gn:
        step_range   = gn[-1][0] - gn[0][0]
        early_cutoff = gn[0][0] + step_range * 0.10
        late          = [(s, v) for s, v in gn if s > early_cutoff and v > 5.0]
        late_gt10     = [(s, v) for s, v in gn if s > early_cutoff and v > 10.0]
        late_gt20     = [(s, v) for s, v in gn if s > early_cutoff and v > 20.0]
        # clustering: spikes within 5 steps of each other
        late_steps = sorted(s for s, _ in late)
        clusters = 0
        i = 0
        while i < len(late_steps):
            j = i + 1
            while j < len(late_steps) and late_steps[j] - late_steps[j-1] <= 5:
                j += 1
            if j - i > 1:
                clusters += 1
            i = j
        d["late_spikes"]       = late
        d["late_spikes_gt10"]  = late_gt10
        d["late_spikes_gt20"]  = late_gt20
        d["spike_max"]         = max((v for _, v in late), default=0.0)
        d["spike_clusters"]    = clusters
        d["early_cutoff"]      = early_cutoff
    else:
        d["late_spikes"] = d["late_spikes_gt10"] = d["late_spikes_gt20"] = []
        d["spike_max"] = 0.0
        d["spike_clusters"] = 0

    # ── clipping saturation ─────────────────────────────────────────────────
    if gnc:
        max_clip = max(v for _, v in gnc)
        d["clip_cap"]         = max_clip
        d["clip_total_steps"] = len(gnc)
        at_cap = sum(1 for _, v in gnc if abs(v - max_clip) < 1e-4)
        d["clip_sat_pct"] = 100.0 * at_cap / len(gnc)
        # per-epoch trend
        boundaries = [s for s, _ in vm] if vm else []
        epoch_sat = []
        prev_b = 0
        for b in boundaries:
            window = [v for s, v in gnc if prev_b < s <= b]
            if window:
                w_sat = sum(1 for v in window if abs(v - max_clip) < 1e-4)
                epoch_sat.append(100.0 * w_sat / len(window))
            prev_b = b
        rest = [v for s, v in gnc if s > prev_b]
        if rest:
            r_sat = sum(1 for v in rest if abs(v - max_clip) < 1e-4)
            epoch_sat.append(100.0 * r_sat / len(rest))
        d["clip_sat_per_epoch"] = epoch_sat
        d["clip_sat_trend"]     = (
            "improving" if len(epoch_sat) >= 2 and epoch_sat[-1] < epoch_sat[0]
            else ("worsening" if len(epoch_sat) >= 2 and epoch_sat[-1] > epoch_sat[0]
                  else "stable")
        )
    else:
        d["clip_sat_pct"] = 0.0
        d["clip_sat_per_epoch"] = []
        d["clip_sat_trend"] = "unknown"

    # ── val mel regression ──────────────────────────────────────────────────
    if vm:
        vm_vals = [v for _, v in vm]
        d["val_mel_series"]    = vm_vals
        d["val_mel_regressed"] = vm_vals[-1] > vm_vals[0] if len(vm_vals) >= 2 else False
        d["val_mel_regressions"] = [
            (i + 2, vm_vals[i+1] - vm_vals[i])
            for i in range(len(vm_vals) - 1) if vm_vals[i+1] > vm_vals[i]
        ]
    else:
        d["val_mel_series"] = []
        d["val_mel_regressed"] = False
        d["val_mel_regressions"] = []

    # ── val/train mel gap ───────────────────────────────────────────────────
    if vm and tm:
        val_dict   = {s: v for s, v in vm}
        train_dict = {s: v for s, v in tm}
        common = sorted(set(val_dict) & set(train_dict))
        gaps = [val_dict[s] - train_dict[s] for s in common]
        d["val_train_gap"]        = gaps[-1] if gaps else None
        d["val_train_gap_series"] = gaps
    else:
        d["val_train_gap"] = None
        d["val_train_gap_series"] = []

    # ── stop token ──────────────────────────────────────────────────────────
    if stop_step:
        p50 = _pct(stop_step, 50)
        bursts = [(s, v) for s, v in stop_step if v > p50 * 2]
        step_range = stop_step[-1][0] - stop_step[0][0]
        late_cut   = stop_step[0][0] + step_range * 0.5
        d["stop_late_bursts"] = [(s, v) for s, v in bursts if s > late_cut]
        d["stop_max"]         = max(v for _, v in stop_step)
    else:
        d["stop_late_bursts"] = []
        d["stop_max"] = 0.0

    if stop_ep and vm:
        stop_dict = {s: v for s, v in stop_ep}
        mel_dict  = {s: v for s, v in vm}
        common    = sorted(set(stop_dict) & set(mel_dict))
        d["stop_mel_ratio"] = stop_dict[common[-1]] / mel_dict[common[-1]] if common else None
    else:
        d["stop_mel_ratio"] = None

    # ── stop/spike co-occurrence ─────────────────────────────────────────────
    # For each grad spike step, check if stop loss was also elevated (±3 steps)
    if stop_step and d["late_spikes"]:
        stop_lookup = {s: v for s, v in stop_step}
        stop_p75    = _pct(stop_step, 75)
        co_occur = []
        for (ss, sv) in d["late_spikes"]:
            for delta in range(-3, 4):
                sv_stop = stop_lookup.get(ss + delta)
                if sv_stop is not None and sv_stop > stop_p75:
                    co_occur.append((ss, sv, sv_stop))
                    break
        d["spike_stop_cooccur"]  = co_occur
        d["spike_stop_cooccur_pct"] = (
            100.0 * len(co_occur) / len(d["late_spikes"]) if d["late_spikes"] else 0.0
        )
    else:
        d["spike_stop_cooccur"]      = []
        d["spike_stop_cooccur_pct"]  = 0.0

    # ── LR phase ────────────────────────────────────────────────────────────
    if dec:
        max_lr  = max(v for _, v in dec)
        curr_lr = dec[-1][1]
        tail    = [v for _, v in dec[-10:]]
        d["lr_max"]         = max_lr
        d["lr_current"]     = curr_lr
        d["lr_still_rising"] = len(tail) >= 2 and tail[-1] > tail[0]
        d["lr_phase"]        = (
            "warmup" if d["lr_still_rising"] and curr_lr >= max_lr * 0.99 else
            ("decay" if curr_lr < max_lr * 0.99 else "peak")
        )
        # spikes during LR ramp vs after
        if gn:
            peak_step = next((s for s, v in dec if abs(v - max_lr) < max_lr * 0.02), None)
            d["lr_peak_step"] = peak_step
            if peak_step:
                d["spikes_during_ramp"] = [(s, v) for s, v in d["late_spikes"] if s <= peak_step]
                d["spikes_post_peak"]   = [(s, v) for s, v in d["late_spikes"] if s > peak_step]
            else:
                d["spikes_during_ramp"] = d["late_spikes"]
                d["spikes_post_peak"]   = []
    else:
        d["lr_phase"] = "unknown"
        d["lr_peak_step"] = None
        d["spikes_during_ramp"] = []
        d["spikes_post_peak"]   = []

    return d
    # store cache (best-effort)
    try:
        if ea is not None:
            setattr(ea, '_diag_cache', d)
    except Exception:
        pass


def tb_print_recommendations(ea):
    """
    Print a prioritized, specific action list based on all diagnostics.
    Only issues that are actually observed generate recommendations.
    """
    print("\n" + "=" * 90)
    print("TENSORBOARD — Analysis & Recommendations")
    print("=" * 90)

    # collect diagnostics once (cached inside _collect_diagnostics)
    d = _collect_diagnostics(ea)
    # Attempt to derive training config from latest checkpoint; fall back to default TrainingConfig
    cfg = None
    try:
        records = load_checkpoints()
        if records:
            last = max(records, key=lambda r: r["epoch_num"])
            cfg = last.get("config")
    except Exception:
        cfg = None

    if cfg is None:
        try:
            from kokoro.training.config import TrainingConfig
            cfg = TrainingConfig()
        except Exception:
            cfg = None

    def cfg_get(attr, default):
        return getattr(cfg, attr, default) if cfg is not None else default

    # Helper suggestions derived from config
    cur_max_lr_mult = cfg_get('max_lr_multiplier', 1.1)
    suggest_max_lr_mult = max(0.1, cur_max_lr_mult - 0.1)
    cur_enc_lr_mult = cfg_get('encoder_lr_multiplier', 1.3)
    suggest_enc_lr_mult = max(0.1, cur_enc_lr_mult - 0.2)
    cur_stop_pos = cfg_get('stop_token_pos_weight', 50.0)
    suggest_stop_pos = max(1.0, int(cur_stop_pos * 0.7))
    cur_stop_loss = cfg_get('stop_token_loss_weight', 0.06)
    suggest_stop_loss = max(0.001, cur_stop_loss * 0.67)
    cur_warmup = cfg_get('warmup_steps', 1200)
    suggest_warmup = int(cur_warmup + 400)
    cur_max_frames = cfg_get('max_frames_per_batch', 15000)
    suggest_max_frames = max(1000, int(cur_max_frames * 0.85))
    cur_weight_decay = cfg_get('weight_decay', 0.01)
    suggest_weight_decay = cur_weight_decay * 2.0
    cur_max_grad = cfg_get('max_grad_norm', 2.0)
    suggest_max_grad = cur_max_grad + 0.5
    recs = []   # list of (priority, label, lines)  priority: 1=CRITICAL 2=WARN 3=INFO

    # ── 1. Val mel overall regression ──────────────────────────────────────
    if d["val_mel_regressed"]:
        body = [
            f"val_mel is higher at the last epoch than the first — training is diverging.",
        ]
        if d["val_train_gap"] is not None and d["val_train_gap"] > 0.8:
            body.append(f"  + Large val/train gap ({d['val_train_gap']:.4f}) suggests overfitting.")
            body.append(f"    → Increase weight_decay (e.g. {cur_weight_decay:.4f} → {suggest_weight_decay:.4f}).")
            body.append(f"    → Increase dropout (e.g. +0.05 to encoder_dropout {cfg_get('encoder_dropout', 0.1):.2f}).")
        if d["clip_sat_pct"] > 40:
            body.append("  + Clipping saturation coincides — gradient pressure is a co-driver.")
            body.append(f"    → Reduce max_lr_multiplier (e.g. {cur_max_lr_mult:.2f} → {suggest_max_lr_mult:.2f}).")
            body.append(f"    → Reduce encoder_lr_multiplier (e.g. {cur_enc_lr_mult:.2f} → {suggest_enc_lr_mult:.2f}).")
        if d["val_mel_regressions"]:
            ep_list = ", ".join(f"Ep{ep}" for ep, _ in d["val_mel_regressions"])
            body.append(f"  Regression epochs: {ep_list}.")
        recs.append((1, "CRITICAL", body))

    # Per-epoch val_mel regression (but not overall)
    elif d["val_mel_regressions"]:
        ep_list = ", ".join(f"Ep{ep} Δ{d:+.4f}" for ep, d in d["val_mel_regressions"])
        recs.append((2, "WARN", [
            f"val_mel regressed at: {ep_list}.",
            "  Monitor closely. If it continues next epoch, reduce max_lr_multiplier by 0.1.",
        ]))

    # ── 2. Grad spikes ─────────────────────────────────────────────────────
    n_late  = len(d["late_spikes"])
    n_gt10  = len(d["late_spikes_gt10"])
    n_gt20  = len(d["late_spikes_gt20"])
    co_pct  = d["spike_stop_cooccur_pct"]
    n_post  = len(d["spikes_post_peak"])
    n_ramp  = len(d["spikes_during_ramp"])

    if n_gt20 > 0:
        body = [
            f"{n_gt20} spike(s) exceeded 20× — severe outlier batches.",
            f"  Max raw grad_norm: {d['spike_max']:.2f}.",
        ]
        if co_pct > 50:
            body.append(f"  {co_pct:.0f}% of spikes co-occur with elevated stop loss.")
            body.append("    → Primary driver is stop BCE bursts.")
            # body.append("    → Reduce stop_token_pos_weight (50 → 35) or stop_token_loss_weight (0.06 → 0.04).")
            body.append("    → Add temporal smoothing to stop targets (soft labels near EOS).")
        # body.append("    → Reduce max_frames in DynamicFrameBatchSampler (15000 → 12000) to cut outlier batch size.")
        # the above recommendation is obsolete
        # we need to create a deep analysis of the outlier batches function and call it here
        # then appent the recommendations it returns to this section
        recs.append((1, "CRITICAL", body))

    elif n_gt10 > 3:
        body = [
            f"{n_gt10} late spike(s) exceeded 10× (max {d['spike_max']:.2f}).",
        ]
        if co_pct > 40:
            body.append(f"  {co_pct:.0f}% co-occur with stop bursts → stop loss is the dominant source.")
            body.append(f"    → Reduce stop_token_pos_weight ({cur_stop_pos:.0f} → {suggest_stop_pos:.0f}).")
        if n_ramp > n_post and d["lr_phase"] in ("warmup", "peak"):
            body.append("  Most spikes during LR ramp → LR pressure is a co-driver.")
            body.append(f"    → Increase warmup_steps ({cur_warmup} → {suggest_warmup}) to soften the ramp slope.")
        body.append(f"    → Lower max_frames ({cur_max_frames} → {suggest_max_frames}) to reduce outlier batch contribution.")
        recs.append((2, "WARN", body))

    elif n_late > 10:
        body = [
            f"{n_late} late spikes >5× — above normal tolerance.",
            f"  max={d['spike_max']:.2f}  clusters={d['spike_clusters']}",
        ]
        if co_pct > 40:
            body.append(f"  {co_pct:.0f}% co-occur with stop bursts.")
            body.append("    → Reduce stop_token_pos_weight slightly (50 → 45).")
        body.append("    → No immediate action needed but watch saturation trend.")
        recs.append((2, "WARN", body))

    elif n_late > 3:
        body = [
            f"{n_late} late spikes >5× — within normal tolerance (threshold for action: >10).",
            f"  max={d['spike_max']:.2f}  clusters={d['spike_clusters']}",
        ]
        if co_pct > 50:
            body.append(f"  {co_pct:.0f}% co-occur with stop bursts — stop loss contributing.")
            body.append("    → Optional: reduce stop_token_pos_weight (50 → 45) if spikes grow next epoch.")
        else:
            body.append("  → No action needed. Continue and re-check after next epoch.")
        recs.append((3, "INFO", body))

    elif n_late > 0:
        recs.append((3, "INFO", [
            f"{n_late} late spike(s) >5× — negligible. No action needed.",
        ]))

    # ── 3. Clipping saturation ─────────────────────────────────────────────
    sat = d["clip_sat_pct"]
    trend = d["clip_sat_trend"]
    if sat > 40 and trend != "improving":
        body = [
            f"Clipping saturation {sat:.1f}% — model is taking truncated steps on nearly half of batches.",
            f"  Trend: {trend}.",
        ]
        body.append(f"    → Reduce max_lr_multiplier (e.g. {cur_max_lr_mult:.2f} → {suggest_max_lr_mult:.2f}) to lower peak gradient pressure.")
        body.append(f"    → Alternatively reduce stop_token_loss_weight ({cur_stop_loss:.4f} → {suggest_stop_loss:.4f}) or pos_weight ({cur_stop_pos:.0f} → {suggest_stop_pos:.0f}) if stop dominates.")
        recs.append((1, "CRITICAL", body))
    elif sat > 20 and trend == "worsening":
        recs.append((2, "WARN", [
            f"Clipping saturation {sat:.1f}% and worsening epoch-over-epoch.",
            f"  Epochs: {[f'{p:.1f}%' for p in d['clip_sat_per_epoch']]}",
            "    → Increase max_grad_norm slightly (e.g. +0.5) to allow learning OR",
            "    → Reduce max_lr_multiplier to lower underlying gradient magnitude.",
        ]))
    elif sat > 20:
        recs.append((3, "INFO", [
            f"Clipping saturation {sat:.1f}% (WARN threshold 20%). Trend: {trend}.",
            "  → Monitor. If still elevated after epoch 3, consider adjusting max_grad_norm.",
        ]))
    elif trend == "improving":
        recs.append((3, "INFO", [
            f"Clipping saturation {sat:.1f}% and improving — no action needed.",
            f"  Per-epoch: {[f'{p:.1f}%' for p in d['clip_sat_per_epoch']]}",
        ]))

    # ── 4. Val/train gap (independent of val regression) ───────────────────
    gap = d["val_train_gap"]
    if gap is not None and gap > 1.0 and not d["val_mel_regressed"]:
        recs.append((2, "WARN", [
            f"Val/train mel gap = {gap:.4f} — overfitting signal.",
            f"    → Increase weight_decay (e.g. {cur_weight_decay:.4f} → {suggest_weight_decay:.4f}).",
            f"    → Increase dropout in transformer layers (+0.05 to encoder_dropout {cfg_get('encoder_dropout', 0.1):.2f}).",
        ]))
    elif gap is not None and gap > 0.5 and not d["val_mel_regressed"]:
        recs.append((3, "INFO", [
            f"Val/train mel gap = {gap:.4f} (mild). Expected during early training.",
            "  → No action yet. Gap typically closes by epoch 4–5.",
        ]))

    # ── 5. Stop token dominance ─────────────────────────────────────────────
    ratio = d["stop_mel_ratio"]
    if ratio is not None and ratio > 0.5:
        body = [
            f"val_stop/val_mel ratio = {ratio:.3f} — stop loss carrying disproportionate weight.",
        ]
        if d["stop_late_bursts"]:
            body.append(f"  Late stop bursts detected: {d['stop_late_bursts'][:4]}")
        body.append(f"    → Reduce stop_token_pos_weight ({cur_stop_pos:.0f} → {suggest_stop_pos:.0f}).")
        body.append(f"    → Or reduce stop_token_loss_weight ({cur_stop_loss:.4f} → {suggest_stop_loss:.4f}).")
        body.append("    → These are multiplicative: effective scale = weight × pos_weight.")
        recs.append((2, "WARN", body))

    # ── 6. LR phase warnings ───────────────────────────────────────────────
    if d["lr_phase"] == "decay" and d["val_mel_regressions"]:
        recs.append((2, "WARN", [
            "Val mel regression detected during LR decay phase.",
            "  This is the most common regression pattern (LR ramp → loss divergence).",
            f"  LR peak was step {d['lr_peak_step']}, regressions at epochs: "
            + ", ".join(f"Ep{ep}" for ep, _ in d["val_mel_regressions"]) + ".",
            f"    → Reduce max_lr_multiplier ({cur_max_lr_mult:.2f} → {suggest_max_lr_mult:.2f}) for the next run.",
            f"    → Or increase pct_start ({cfg_get('pct_start', 0.2):.2f} → {min(0.5, cfg_get('pct_start', 0.2) + 0.05):.2f}) to spend more time in warmup.",
        ]))

    # ── 7. All clear ────────────────────────────────────────────────────────
    critical = [r for r in recs if r[0] == 1]
    warns    = [r for r in recs if r[0] == 2]
    infos    = [r for r in recs if r[0] == 3]

    if not critical and not warns:
        print("  ✓  No action items. Training is progressing normally.")
        if infos:
            print()
    elif critical:
        print(f"  {len(critical)} CRITICAL  {len(warns)} WARN  {len(infos)} INFO\n")
    else:
        print(f"  {len(warns)} WARN  {len(infos)} INFO\n")

    priority_labels = {1: "CRITICAL", 2: "WARN    ", 3: "INFO    "}
    icons           = {1: "✗", 2: "⚠", 3: "ℹ"}

    for idx, (pri, label, lines) in enumerate(recs, 1):
        icon = icons[pri]
        print(f"  [{idx}] {icon} {priority_labels[pri]}  {lines[0]}")
        for line in lines[1:]:
            print(f"            {line}")
        print()


def collect_param_drills(ea, records=None, force=False):
    """
    Lazy, cached runner for expensive per-parameter analyses.
    Returns a dict of drill results (e.g., spike cause analysis) and caches
    results on `ea` so repeated calls are cheap.
    """
    try:
        if ea is not None and hasattr(ea, '_param_drill_cache') and not force:
            return getattr(ea, '_param_drill_cache')
    except Exception:
        pass

    res = {}
    # Expensive drill: gradient pre-clip pressure analysis
    try:
        # reuse tb_print_spike_cause_analysis internals by calling it in a dry-run
        # but to avoid printing, replicate minimal computation here.
        # We compute only the summary counts which are frequently useful.
        if records and len(records) >= 2:
            rec_b = records[-1]
            rec_a = records[-2]
            ps = rec_b.get('param_stats', {})
            n_with_delta = sum(1 for v in ps.values() if v.get('delta') is not None)
            res['param_delta_count'] = n_with_delta
        else:
            res['param_delta_count'] = 0
    except Exception:
        res['param_delta_count'] = 0

    try:
        if ea is not None:
            setattr(ea, '_param_drill_cache', res)
    except Exception:
        pass

    return res


def tb_print_val_mel_series(ea):
    """Per-epoch val mel with explicit Δ and regression flags."""
    vm = _get(ea, "loss/val_mel_epoch")
    if not vm:
        return
    print("\n" + "=" * 90)
    print("TENSORBOARD — Val Mel Epoch Series")
    print("=" * 90)
    prev = None
    for i, (s, v) in enumerate(vm):
        if prev is None:
            flag = ""
        elif v > prev:
            flag = f"  ▲ +{v - prev:.5f}  ← REGRESSION"
        else:
            flag = f"  ▼ {v - prev:+.5f}"
        print(f"  Ep{i+1:02d}  step={s:>5}  val_mel={v:.5f}{flag}")
        prev = v
    # Summary
    if len(vm) >= 2:
        vals = [v for _, v in vm]
        best = min(vals)
        best_ep = vals.index(best) + 1
        print(f"\n  best={best:.5f} at Ep{best_ep:02d}  "
              f"total Δ={vals[-1] - vals[0]:+.5f}  "
              f"last={vals[-1]:.5f}")


def tb_print_mel_stop_window_correlation(ea):
    """
    200-step windowed table of mel mean, stop mean, and LR% side by side.
    Makes it easy to see whether stop and mel move together (LR-driven)
    or stop leads mel (stop-token source).
    """
    mel  = _get(ea, "loss/mel")
    stop = _get(ea, "loss/stop")
    lr   = _get(ea, "stats/lr_decoder")

    if not mel:
        return

    print("\n" + "=" * 90)
    print("TENSORBOARD — Mel vs Stop Loss Correlation (200-step windows)")
    print("=" * 90)
    print(f"  {'Window':<12}  {'mel mean':>9}  {'Δmel':>6}  "
          f"{'stop mean':>9}  {'Δstop':>6}  {'LR%':>6}  co-move?")
    print("  " + "-" * 72)

    max_step  = mel[-1][0] if mel else 0
    lr_max    = max(v for _, v in lr) if lr else 1.0
    lr_lookup = {s: v for s, v in lr}

    prev_mel_m = prev_stop_m = None
    w_start = 0
    w_size  = 200

    # find first window that has data
    first_step = mel[0][0]
    w_start = (first_step // w_size) * w_size

    while w_start <= max_step:
        w_end = w_start + w_size
        seg_mel  = [v for s, v in mel  if w_start <= s < w_end]
        seg_stop = [v for s, v in stop if w_start <= s < w_end] if stop else []
        # LR: nearest sample at midpoint
        mid = w_start + w_size // 2
        lr_here = min(lr_lookup.items(), key=lambda x: abs(x[0] - mid))[1] if lr_lookup else 0
        lr_pct  = 100.0 * lr_here / lr_max if lr_max else 0

        if seg_mel:
            mm = statistics.mean(seg_mel)
            sm = statistics.mean(seg_stop) if seg_stop else None

            dmel  = mm - prev_mel_m  if prev_mel_m  is not None else None
            dstop = (sm - prev_stop_m) if (prev_stop_m is not None and sm is not None) else None

            mel_arr  = ("▲" if dmel  and dmel  > 0 else "▼") if dmel  is not None else " "
            stop_arr = ("▲" if dstop and dstop > 0 else "▼") if dstop is not None else " "

            # co-movement: both up or both down
            comove = ""
            if dmel is not None and dstop is not None:
                if (dmel > 0 and dstop > 0):
                    comove = "both↑ (LR pressure)"
                elif (dmel < 0 and dstop < 0):
                    comove = "both↓ (improving)"
                elif dstop > 0 and dmel <= 0:
                    comove = "stop↑ only (stop source)"
                elif dmel > 0 and dstop <= 0:
                    comove = "mel↑ only"

            dmel_str  = f"{dmel:+.4f}" if dmel  is not None else "      "
            dstop_str = f"{dstop:+.4f}" if dstop is not None else "      "
            sm_str    = f"{sm:.5f}" if sm is not None else "       ?"

            print(f"  {w_start:>5}–{w_end:<5}  {mm:>9.5f} {mel_arr} {dmel_str}  "
                  f"{sm_str} {stop_arr} {dstop_str}  {lr_pct:>5.1f}%  {comove}")

            prev_mel_m  = mm
            prev_stop_m = sm

        w_start = w_end


def tb_print_lr_phase_detail(ea, records=None):
    """
    Detailed LR table from 90% of observed LR onward.
    % columns are relative to the TRUE schedule peak derived from the saved
    config (learning_rate × max_lr_multiplier at step warmup_steps +
    pct_start × onecycle_steps), so you can see how far into the full
    training arc you are — not just how far above the current-session max.
    Falls back to observed max when config is unavailable.
    """
    lr  = _get(ea, "stats/lr_decoder")
    enc = _get(ea, "stats/lr_encoder")
    if not lr:
        return

    # ── Derive true schedule peak from the saved config ───────────────────
    true_peak_dec = None
    true_peak_enc = None
    abs_peak_step = None
    cfg_note      = "(config unavailable — falling back to observed max)"
    if records:
        last = max(records, key=lambda r: r["epoch_num"])
        cfg  = last.get("config")
        if cfg is not None:
            learning_rate = getattr(cfg, 'learning_rate',              1e-4)
            max_lr_mult   = getattr(cfg, 'max_lr_multiplier',          1.1)
            enc_lr_mult   = getattr(cfg, 'encoder_lr_multiplier',      1.3)
            pct_start     = getattr(cfg, 'pct_start',                  0.2)
            use_warmup    = getattr(cfg, 'use_warmup',                  True)
            warmup_steps  = getattr(cfg, 'warmup_steps',               1200) if use_warmup else 0
            num_epochs    = getattr(cfg, 'num_epochs',                  100)
            # Derive steps_per_epoch from latest checkpoint: optimizer_steps / epoch_num
            opt_steps = last.get("optimizer_steps")
            ep_num    = last.get("epoch_num")
            if isinstance(opt_steps, int) and isinstance(ep_num, int) and ep_num > 0:
                steps_per_epoch = opt_steps // ep_num
                total_steps     = num_epochs * steps_per_epoch
                # Replicate _apply_warmup_guard
                if warmup_steps >= total_steps:
                    warmup_steps = max(0, total_steps - 1)
                onecycle_steps  = max(1, total_steps - warmup_steps)
                peak_onecycle   = int(pct_start * onecycle_steps)
                abs_peak_step   = warmup_steps + peak_onecycle
                true_peak_dec   = learning_rate * max_lr_mult
                true_peak_enc   = learning_rate * max_lr_mult * enc_lr_mult
                cfg_note        = (
                    f"lr={learning_rate:.2e}  ×  max_lr_mult={max_lr_mult}  "
                    f"steps/epoch={steps_per_epoch}  warmup={warmup_steps}  "
                    f"onecycle={onecycle_steps}  pct_start={pct_start}"
                )

    # ── Observed data ──────────────────────────────────────────────────────
    obs_max_dec = max(v for _, v in lr)
    curr_lr     = lr[-1][1]
    curr_step   = lr[-1][0]

    # Reference peak for all % calculations
    ref_peak_dec = true_peak_dec if true_peak_dec is not None else obs_max_dec
    ref_peak_enc = true_peak_enc if true_peak_enc is not None else (max(v for _, v in enc) if enc else ref_peak_dec)

    # Phase: is LR still rising?
    tail = [v for _, v in lr[-10:]]
    still_rising = len(tail) >= 2 and tail[-1] > tail[0]
    # Past-peak step (only relevant when not still rising)
    if not still_rising:
        obs_peak_step = next((s for s, v in lr if v >= obs_max_dec * 0.99), None)
    else:
        obs_peak_step = None

    print("\n" + "=" * 90)
    print("TENSORBOARD — LR Phase Detail (from 90% of peak onward)")
    print("=" * 90)
    print(f"  schedule      : {cfg_note}")

    if true_peak_dec is not None:
        steps_to_peak = abs_peak_step - curr_step
        direction     = f"{steps_to_peak:+d} steps to peak" if steps_to_peak > 0 else f"{-steps_to_peak} steps past peak"
        print(f"  true peak LR  : {true_peak_dec:.8f} (dec)   {true_peak_enc:.8f} (enc)   at step {abs_peak_step}  [{direction}]")
    else:
        print(f"  obs  max LR   : {obs_max_dec:.8f}")

    pct_now = 100.0 * curr_lr / ref_peak_dec
    print(f"  curr LR (dec) : {curr_lr:.8f}  at step {curr_step}  ({pct_now:.3f}% of true peak)")

    if still_rising:
        phase = "WARMUP → approaching peak"
    elif curr_lr >= ref_peak_dec * 0.99:
        phase = "AT / NEAR PEAK"
    elif curr_lr >= ref_peak_dec * 0.80:
        phase = f"EARLY DECAY ({100 * curr_lr / ref_peak_dec:.1f}% of peak)"
    else:
        phase = f"DECAY ({100 * curr_lr / ref_peak_dec:.1f}% of peak)"
    print(f"  phase         : {phase}")

    # ── Step-by-step table ─────────────────────────────────────────────────
    # Start from the step where observed LR first hit 90% of observed max.
    thresh_90_step = next((s for s, v in lr if v >= obs_max_dec * 0.90), lr[0][0])

    print(f"\n  Step-by-step from step {thresh_90_step} onward  (% = fraction of true schedule peak):")
    print(f"  {'step':>6}  {'lr_decoder':>12}  {'% true peak':>12}  {'lr_encoder':>12}  {'% true peak':>12}")
    print("  " + "-" * 66)

    enc_lookup  = {s: v for s, v in enc} if enc else {}
    # Flag the bucket that contains the true absolute peak step (config-derived).
    # Fall back to observed peak bucket when config isn't available.
    flag_bucket = None
    if abs_peak_step is not None:
        flag_bucket = (abs_peak_step // 100) * 100
    elif obs_peak_step is not None and not still_rising:
        flag_bucket = (obs_peak_step // 100) * 100

    seen_buckets = set()
    for s, v in lr:
        if s < thresh_90_step:
            continue
        bucket = (s // 100) * 100
        if bucket in seen_buckets:
            continue
        seen_buckets.add(bucket)
        dec_pct = 100.0 * v / ref_peak_dec
        enc_v   = enc_lookup.get(s)
        enc_str = f"{enc_v:.8f}  {100*enc_v/ref_peak_enc:>11.3f}%" if enc_v else "              ?             ?"
        flag    = "  ← peak" if (flag_bucket is not None and bucket == flag_bucket) else ""
        print(f"  {s:>6}  {v:.8f}  {dec_pct:>11.3f}%{flag}  {enc_str}")

    # When still in warmup, append a projected-peak row so the table endpoint is visible.
    if still_rising and abs_peak_step is not None and abs_peak_step > curr_step:
        proj_bucket = (abs_peak_step // 100) * 100
        if proj_bucket not in seen_buckets:
            enc_proj = f"{true_peak_enc:.8f}  {'100.000%':>11}" if true_peak_enc else "?"
            print(f"  {abs_peak_step:>6}  {true_peak_dec:.8f}  {'100.000%':>11}  ← peak (projected)  {enc_proj}")


def tb_print_late_spike_context(ea):
    """
    For every grad spike after the first 10% of training steps,
    print its LR (as % of peak) and the nearest stop loss value.
    Makes it easy to attribute each spike to LR pressure vs stop burst.
    """
    gn   = _get(ea, "stats/grad_norm")
    stop = _get(ea, "loss/stop")
    lr   = _get(ea, "stats/lr_decoder")
    if not gn:
        return

    step_range   = gn[-1][0] - gn[0][0]
    early_cutoff = gn[0][0] + step_range * 0.10
    late_spikes  = [(s, v) for s, v in gn if s > early_cutoff and v > 5.0]

    if not late_spikes:
        return

    print("\n" + "=" * 90)
    print(f"TENSORBOARD — Late Grad Spike Context ({len(late_spikes)} spikes >5.0 after init)")
    print("=" * 90)

    lr_max    = max(v for _, v in lr) if lr else 1.0
    lr_lookup = {s: v for s, v in lr} if lr else {}
    stop_list = stop if stop else []

    # stop p75 for "elevated" threshold
    stop_p75 = _pct(stop, 75) if stop else 0.0

    print(f"  {'step':>6}  {'raw grad':>9}  {'lr %peak':>9}  {'stop nearby':>12}  {'stop elevated?':>15}  attribution")
    print("  " + "-" * 80)

    for s, v in sorted(late_spikes):
        lr_here  = min(lr_lookup.items(), key=lambda x: abs(x[0] - s))[1] if lr_lookup else 0
        lr_pct   = 100.0 * lr_here / lr_max if lr_max else 0
        stop_here = min(stop_list, key=lambda x: abs(x[0] - s))[1] if stop_list else 0
        stop_flag = "YES ▲" if stop_here > stop_p75 else "no"

        # attribution heuristic
        if stop_here > stop_p75 and lr_pct < 97:
            attr = "stop burst"
        elif lr_pct >= 97 and stop_here <= stop_p75:
            attr = "LR at peak"
        elif lr_pct >= 97 and stop_here > stop_p75:
            attr = "LR peak + stop"
        else:
            attr = "outlier batch"

        print(f"  {s:>6}  {v:>9.3f}  {lr_pct:>8.1f}%  {stop_here:>12.5f}  {stop_flag:>15}  {attr}")

    # summary
    attrs = []
    for s, v in late_spikes:
        lr_here  = min(lr_lookup.items(), key=lambda x: abs(x[0] - s))[1] if lr_lookup else 0
        lr_pct   = 100.0 * lr_here / lr_max if lr_max else 0
        stop_here = min(stop_list, key=lambda x: abs(x[0] - s))[1] if stop_list else 0
        if stop_here > stop_p75 and lr_pct < 97:
            attrs.append("stop")
        elif lr_pct >= 97 and stop_here <= stop_p75:
            attrs.append("lr")
        elif lr_pct >= 97 and stop_here > stop_p75:
            attrs.append("both")
        else:
            attrs.append("batch")
    counts = {k: attrs.count(k) for k in set(attrs)}
    print(f"\n  Attribution summary: " + "  ".join(f"{k}={n}" for k, n in sorted(counts.items())))


def tb_print_spike_cause_analysis(ea, records=None):
    """
    Estimates per-tensor gradient pressure from consecutive checkpoint weight deltas.

    For each parameter tensor in the most recent checkpoint pair, computes:
        avg_grad ≈ weight_delta / (epoch_steps × lr)

    and compares it to the tensor's pre-clip cap (as configured in trainer.py).
    Groups results by pre-clip category, computes worst-case theoretical co-spike
    norm (√Σcap²), and flags groups that are near or saturating their caps.

    This makes it possible to see *structurally* which parameter groups drive
    observed grad spikes — and whether any groups are effectively uncapped.
    """
    from collections import defaultdict

    if not records or len(records) < 2:
        return

    # ── Find last checkpoint pair with valid weight deltas ─────────────────
    rec_b = rec_a = None
    for i in range(len(records) - 1, 0, -1):
        ps = records[i].get("param_stats", {})
        if any(v.get("delta") is not None for v in ps.values()):
            rec_b = records[i]
            rec_a = records[i - 1]
            break
    if rec_b is None:
        return

    steps_b = rec_b.get("optimizer_steps")
    steps_a = rec_a.get("optimizer_steps")
    if not (isinstance(steps_b, int) and isinstance(steps_a, int) and steps_b > steps_a):
        return
    ep_steps = steps_b - steps_a

    # ── Average LR for this epoch from TB ─────────────────────────────────
    lr_series  = _get(ea, "stats/lr_decoder")
    enc_series = _get(ea, "stats/lr_encoder")

    if lr_series:
        in_range_dec = [v for s, v in lr_series if steps_a <= s <= steps_b]
        avg_lr_dec = statistics.mean(in_range_dec) if in_range_dec else max(v for _, v in lr_series)
    else:
        return  # can't compute avg_grad without LR

    if enc_series:
        in_range_enc = [v for s, v in enc_series if steps_a <= s <= steps_b]
        avg_lr_enc = statistics.mean(in_range_enc) if in_range_enc else None
    else:
        avg_lr_enc = None

    cfg = rec_b.get("config")
    enc_lr_mult = getattr(cfg, 'encoder_lr_multiplier', 1.3) if cfg else 1.3
    if avg_lr_enc is None:
        avg_lr_enc = avg_lr_dec * enc_lr_mult

    # ── Pre-clip cap values from config (fall back to trainer.py defaults) ─
    if cfg is not None:
        dec_ffn_cap = getattr(cfg, 'ffn_spike_clip_norm',         8.0)
        enc_ffn_cap = getattr(cfg, 'encoder_ffn_spike_clip_norm', 12.0)
        attn_cap    = getattr(cfg, 'attention_spike_clip_norm',   20.0)
        proj_cap    = getattr(cfg, 'projection_spike_clip_norm',  20.0)
        stop_cap    = getattr(cfg, 'stop_head_spike_clip_norm',    1.0)
    else:
        dec_ffn_cap = 8.0
        enc_ffn_cap = 12.0
        attn_cap    = 20.0
        proj_cap    = 20.0
        stop_cap    = 1.0

    # ── Classify each tensor → (category, cap, lr_for_tensor) ─────────────
    def classify(name):
        if re.search(r'decoder\.layers\.\d+\.(linear1|linear2)', name):
            return ('dec_ffn',  dec_ffn_cap, avg_lr_dec)
        if re.search(r'transformer_encoder_layers\.\d+\.(linear1|linear2)', name):
            return ('enc_ffn',  enc_ffn_cap, avg_lr_enc)
        if re.search(r'decoder\.layers\.\d+\.(self_attn|cross_attn)\.(w_q|w_k|w_v|w_o)', name):
            return ('dec_attn', attn_cap,    avg_lr_dec)
        if re.search(r'transformer_encoder_layers\.\d+\.self_attn\.(w_q|w_k|w_v|w_o)', name):
            return ('enc_attn', attn_cap,    avg_lr_enc)
        if re.search(r'(mel_projection|mel_linear)', name):
            return ('mel_proj', proj_cap,    avg_lr_dec)
        if re.search(r'stop_token_predictor', name):
            return ('stop',     stop_cap,    avg_lr_dec)
        return ('other', float('inf'), avg_lr_dec)

    # ── Collect avg_grad per tensor ────────────────────────────────────────
    groups = defaultdict(list)  # cat → list of (name, avg_grad, cap)
    ps = rec_b.get("param_stats", {})
    for name, s in ps.items():
        delta = s.get("delta")
        if delta is None:
            continue
        cat, cap, lr_t = classify(name)
        if not lr_t or lr_t <= 0:
            continue
        avg_grad = delta / (ep_steps * lr_t)
        groups[cat].append((name, avg_grad, cap))

    if not groups:
        return

    # ── Observed max late spike ────────────────────────────────────────────
    gn = _get(ea, "stats/grad_norm")
    if gn:
        step_range   = gn[-1][0] - gn[0][0]
        early_cutoff = gn[0][0] + step_range * 0.10
        obs_max      = max((v for s, v in gn if s > early_cutoff), default=0.0)
    else:
        obs_max = 0.0

    # ── Print ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TENSORBOARD — Gradient Pre-Clip Pressure Analysis")
    print("=" * 90)
    print(f"  Checkpoint pair : {rec_a['file']}  →  {rec_b['file']}")
    print(f"  Epoch steps     : {ep_steps}")
    print(f"  Avg LR decoder  : {avg_lr_dec:.8f}   encoder: {avg_lr_enc:.8f}")
    print(f"  Pre-clip caps   : dec_ffn={dec_ffn_cap}  enc_ffn={enc_ffn_cap}  "
          f"attn={attn_cap}  mel_proj={proj_cap}  stop={stop_cap}")

    CAT_LABELS = {
        'dec_ffn':  'Decoder  FFN',
        'enc_ffn':  'Encoder  FFN',
        'dec_attn': 'Decoder Attn',
        'enc_attn': 'Encoder Attn',
        'mel_proj': 'Mel Proj    ',
        'stop':     'Stop Head   ',
        'other':    'Other(uncpd) ',
    }
    CAT_ORDER = ['dec_ffn', 'enc_ffn', 'dec_attn', 'enc_attn', 'mel_proj', 'stop', 'other']

    print()
    print(f"  {'Category':<14} {'N':>4}  {'min_g':>8}  {'avg_g':>8}  {'max_g':>8}  "
          f"{'cap':>6}  {'≥50%cap':>8}  {'≥cap':>6}  {'√(N·cap²)':>10}  flags")
    print("  " + "-" * 90)

    theoretical_max_sq = 0.0
    flags_list = []

    for cat in CAT_ORDER:
        if cat not in groups:
            continue
        entries = groups[cat]
        grads   = [g for _, g, _ in entries]
        cap     = entries[0][2]
        n       = len(entries)
        min_g   = min(grads)
        avg_g   = statistics.mean(grads)
        max_g   = max(grads)
        near    = sum(1 for g in grads if g >= cap * 0.5)   # ≥ 50% of cap
        sat     = sum(1 for g in grads if g >= cap)          # at or above cap

        if cap != float('inf'):
            cosp = math.sqrt(n * cap ** 2)
            theoretical_max_sq += n * cap ** 2
            cosp_str = f"±{cosp:>8.1f}"
        else:
            # For uncapped tensors, use actual avg_grad as a proxy
            cosp = math.sqrt(sum(g ** 2 for g in grads))
            cosp_str = f"~{cosp:>8.1f}"

        flag = ""
        if cap == float('inf') and max_g > 5.0:
            flag = "UNCAPPED+HIGH"
            flags_list.append(
                f"  {CAT_LABELS.get(cat, cat)}: {n} uncapped tensors  max avg_grad={max_g:.2f}")
        elif sat > 0:
            flag = f"AT CAP ({sat}/{n})"
            flags_list.append(
                f"  {CAT_LABELS.get(cat, cat)}: {sat}/{n} tensors avg_grad ≥ cap  "
                f"(avg={avg_g:.2f}  cap={cap})")
        elif near > n // 2:
            flag = "NEAR CAP"
            flags_list.append(
                f"  {CAT_LABELS.get(cat, cat)}: {near}/{n} tensors avg_grad ≥ 50% of cap  "
                f"(avg={avg_g:.2f}  cap={cap})")

        cap_str = f"{cap:.1f}" if cap != float('inf') else "  ∞"
        print(f"  {CAT_LABELS.get(cat, cat):<14} {n:>4}  {min_g:>8.3f}  {avg_g:>8.3f}  "
              f"{max_g:>8.3f}  {cap_str:>6}  {near:>4}/{n:<3}  {sat:>3}/{n:<3}  "
              f"{cosp_str}  {flag}")

    theoretical_max = math.sqrt(theoretical_max_sq)
    print()
    print(f"  Theoretical worst-case co-spike  √(Σ N·cap²) : {theoretical_max:.1f}")
    if obs_max > 0:
        print(f"  Observed max late spike in TB               : {obs_max:.2f}")
        print(f"  Headroom (theory / observed)                : {theoretical_max / obs_max:.1f}×"
              "  (< 2× means pre-clips are the binding constraint)")

    if flags_list:
        print(f"\n  Flagged groups:")
        for fl in flags_list:
            print(fl)
    else:
        print(f"\n  All groups below 50% of their pre-clip caps — gradient pressure nominal.")


def tb_analyze(log_dir: Path = TB_LOG_DIR, records=None):
    ea = _load_tb(log_dir)
    if ea is None:
        return
    print(f"\n[TensorBoard] Log dir: {log_dir}")
    tags = sorted(ea.Tags().get("scalars", []))
    print(f"[TensorBoard] {len(tags)} scalar tags available: {tags}")

    tb_print_step_loss_summary(ea)
    tb_print_val_mel_series(ea)
    tb_print_epoch_table(ea)
    tb_print_mel_stop_window_correlation(ea)
    tb_print_stop_token_analysis(ea)
    tb_print_gradient_analysis(ea)
    tb_print_late_spike_context(ea)
    tb_print_spike_cause_analysis(ea, records=records)
    tb_print_lr_trajectory(ea)
    tb_print_lr_phase_detail(ea, records=records)
    tb_print_regression_flags(ea)
    tb_print_recommendations(ea)


def main():
    global CHECKPOINT_DIR, TB_LOG_DIR

    parser = argparse.ArgumentParser(
        description="Checkpoint + TensorBoard regression analysis."
    )
    parser.add_argument(
        "--model",
        default=str(CHECKPOINT_DIR),
        metavar="DIR",
        help="Path to the model directory containing checkpoints and a logs/ sub-folder "
             f"(default: {CHECKPOINT_DIR})",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR = Path(args.model)
    TB_LOG_DIR     = CHECKPOINT_DIR / "logs"

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

    # ── TensorBoard section ───────────────────────────────────────────────────
    tb_analyze(TB_LOG_DIR, records=records)


if __name__ == "__main__":
    main()
