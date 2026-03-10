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


def tb_print_recommendations(ea):
    """
    Print a prioritized, specific action list based on all diagnostics.
    Only issues that are actually observed generate recommendations.
    """
    print("\n" + "=" * 90)
    print("TENSORBOARD — Analysis & Recommendations")
    print("=" * 90)

    d = _collect_diagnostics(ea)
    recs = []   # list of (priority, label, lines)  priority: 1=CRITICAL 2=WARN 3=INFO

    # ── 1. Val mel overall regression ──────────────────────────────────────
    if d["val_mel_regressed"]:
        body = [
            f"val_mel is higher at the last epoch than the first — training is diverging.",
        ]
        if d["val_train_gap"] is not None and d["val_train_gap"] > 0.8:
            body.append(f"  + Large val/train gap ({d['val_train_gap']:.4f}) suggests overfitting.")
            body.append("    → Increase weight_decay (try 1e-3 → 2e-3).")
            body.append("    → Increase dropout (e.g. +0.05 in transformer layers).")
        if d["clip_sat_pct"] > 40:
            body.append("  + Clipping saturation coincides — gradient pressure is a co-driver.")
            body.append("    → Reduce max_lr_multiplier (e.g. 1.2 → 1.0).")
            body.append("    → Reduce encoder_lr_multiplier (e.g. 1.3 → 1.1).")
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
            body.append("    → Reduce stop_token_pos_weight (50 → 35) or stop_token_loss_weight (0.06 → 0.04).")
            body.append("    → Add temporal smoothing to stop targets (soft labels near EOS).")
        body.append("    → Reduce max_frames in DynamicFrameBatchSampler (15000 → 12000) to cut outlier batch size.")
        recs.append((1, "CRITICAL", body))

    elif n_gt10 > 3:
        body = [
            f"{n_gt10} late spike(s) exceeded 10× (max {d['spike_max']:.2f}).",
        ]
        if co_pct > 40:
            body.append(f"  {co_pct:.0f}% co-occur with stop bursts → stop loss is the dominant source.")
            body.append("    → Reduce stop_token_pos_weight (50 → 40).")
        if n_ramp > n_post and d["lr_phase"] in ("warmup", "peak"):
            body.append("  Most spikes during LR ramp → LR pressure is a co-driver.")
            body.append("    → Increase warmup_steps (1200 → 1600) to soften the ramp slope.")
        body.append("    → Lower max_frames (15000 → 13000) to reduce outlier batch contribution.")
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
        body.append("    → Reduce max_lr_multiplier (e.g. 1.2 → 1.0) to lower peak gradient pressure.")
        body.append("    → Alternatively reduce stop_token_loss_weight or pos_weight if stop dominates.")
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
            "    → Increase weight_decay (e.g. 1e-3 → 2e-3).",
            "    → Increase dropout in transformer layers (+0.05).",
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
        body.append("    → Reduce stop_token_pos_weight (50 → 35–40).")
        body.append("    → Or reduce stop_token_loss_weight (0.06 → 0.04).")
        body.append("    → These are multiplicative: effective scale = weight × pos_weight.")
        recs.append((2, "WARN", body))

    # ── 6. LR phase warnings ───────────────────────────────────────────────
    if d["lr_phase"] == "decay" and d["val_mel_regressions"]:
        recs.append((2, "WARN", [
            "Val mel regression detected during LR decay phase.",
            "  This is the most common regression pattern (LR ramp → loss divergence).",
            f"  LR peak was step {d['lr_peak_step']}, regressions at epochs: "
            + ", ".join(f"Ep{ep}" for ep, _ in d["val_mel_regressions"]) + ".",
            "    → Reduce max_lr_multiplier (1.2 → 1.1) for the next run.",
            "    → Or increase pct_start (0.2 → 0.25) to spend more time in warmup.",
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


def tb_analyze(log_dir: Path = TB_LOG_DIR):
    ea = _load_tb(log_dir)
    if ea is None:
        return
    print(f"\n[TensorBoard] Log dir: {log_dir}")
    tags = sorted(ea.Tags().get("scalars", []))
    print(f"[TensorBoard] {len(tags)} scalar tags available: {tags}")

    tb_print_step_loss_summary(ea)
    tb_print_epoch_table(ea)
    tb_print_stop_token_analysis(ea)
    tb_print_gradient_analysis(ea)
    tb_print_lr_trajectory(ea)
    tb_print_regression_flags(ea)
    tb_print_recommendations(ea)


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

    # ── TensorBoard section ───────────────────────────────────────────────────
    tb_analyze()


if __name__ == "__main__":
    main()
