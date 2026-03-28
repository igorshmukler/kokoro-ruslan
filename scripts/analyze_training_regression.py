"""
Checkpoint regression analysis script.
For each checkpoint, extracts:
  - epoch, loss, optimizer_steps_completed, ema_updates
  - weight norm / mean / std / max-abs for every named parameter
  - weight change (L2 norm delta) from the previous checkpoint
  - EMA vs live weight divergence per checkpoint
  - any NaN or Inf weights
Prints a sorted summary table and flags gradual degradation, sudden spikes,
and per-layer instability across the full run.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
import re
import math
import statistics
from pathlib import Path
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


CHECKPOINT_DIR = Path("my_model")
TB_LOG_DIR = Path("my_model/logs")

KEY_LAYERS = [
    "mel_projection",
    "mel_linear",
    "duration_predictor.linear",
    "pitch_predictor.linear",
    "energy_predictor.linear",
    "stop_token_predictor",
    "decoder.layers.0.self_attn.w_o",
    "decoder.layers.0.ff.linear1",
    "decoder.layers.0.ff.linear2",
    "decoder.layers.5.self_attn.w_o",
    "text_embedding",
]


# ─────────────────────────────────────────────────────────────────────────────
# Shared parameter classifier  (was duplicated in two functions — now central)
# ─────────────────────────────────────────────────────────────────────────────

def classify_param(name: str, cfg=None):
    """
    Return (category, pre_clip_cap, lr_multiplier_key) for a parameter name.

    cap is the per-element gradient-norm cap used by the trainer's pre-clip
    logic (or float('inf') for truly uncapped groups).
    lr_multiplier_key is 'decoder' or 'encoder'.

    Category coverage is intentionally broad so that the 'other' residual is
    small and any entry in it is genuinely novel/unknown.
    """
    if cfg is not None:
        dec_ffn_cap = getattr(cfg, 'ffn_spike_clip_norm',              8.0)
        enc_ffn_cap = getattr(cfg, 'encoder_ffn_spike_clip_norm',     12.0)
        attn_cap    = getattr(cfg, 'attention_spike_clip_norm',        20.0)
        proj_cap    = getattr(cfg, 'projection_spike_clip_norm',       20.0)
        stop_cap    = getattr(cfg, 'stop_head_spike_clip_norm',         0.5)
    else:
        dec_ffn_cap = 8.0
        enc_ffn_cap = 12.0
        attn_cap    = 20.0
        proj_cap    = 20.0
        stop_cap    = 1.0

    # ── Decoder FFN ──────────────────────────────────────────────────────────
    # Matches both direct children (.linear1) and sub-module children (.ff.linear1)
    # lr_key = 'decoder_ffn' so the dedicated param group LR is used when available.
    if re.search(r'decoder\.layers\.\d+\.(?:ff\.)?(linear1|linear2)', name):
        return ('dec_ffn',  dec_ffn_cap, 'decoder_ffn')
    # ── Encoder FFN ──────────────────────────────────────────────────────────
    if re.search(r'transformer_encoder_layers\.\d+\.(?:ff\.)?(linear1|linear2)', name):
        return ('enc_ffn',  enc_ffn_cap, 'encoder')
    # ── Decoder attention ─────────────────────────────────────────────────────
    if re.search(r'decoder\.layers\.\d+\.(self_attn|cross_attn)\.(w_q|w_k|w_v|w_o)', name):
        return ('dec_attn', attn_cap,    'decoder')
    # ── Encoder attention ─────────────────────────────────────────────────────
    if re.search(r'transformer_encoder_layers\.\d+\.self_attn\.(w_q|w_k|w_v|w_o)', name):
        return ('enc_attn', attn_cap,    'encoder')
    # ── Mel projection / output head ─────────────────────────────────────────
    if re.search(r'(mel_projection|mel_linear)', name):
        return ('mel_proj', proj_cap,    'decoder')
    # ── Stop token head ──────────────────────────────────────────────────────
    if re.search(r'stop_token', name):
        return ('stop',     stop_cap,    'decoder')
    # ── Variance adaptors (duration / pitch / energy predictors) ─────────────
    if re.search(r'(duration_predictor|pitch_predictor|energy_predictor)', name):
        return ('variance', proj_cap,    'decoder')
    # ── Text / phoneme embeddings ─────────────────────────────────────────────
    if re.search(r'(text_embedding|phoneme_embedding|embedding)', name):
        return ('embedding', float('inf'), 'encoder')
    # ── Layer normalisation (weights + biases) ────────────────────────────────
    if re.search(r'(layer_norm|layernorm|norm)\d*\.(weight|bias)$', name, re.IGNORECASE):
        return ('layer_norm', float('inf'), 'decoder')
    # ── All bias terms (not already caught above) ─────────────────────────────
    if name.endswith('.bias'):
        return ('bias', float('inf'), 'decoder')
    # ── Convolutional layers ──────────────────────────────────────────────────
    if re.search(r'\bconv\b|\bconv1d\b|\bconv2d\b', name, re.IGNORECASE):
        return ('conv', proj_cap, 'decoder')
    # ── Positional encodings (fixed buffers — should not move) ───────────────
    if re.search(r'(pos_enc|positional_encoding|pe\.weight)', name, re.IGNORECASE):
        return ('pos_enc', float('inf'), 'decoder')
    # ── Catch-all for everything genuinely unrecognised ───────────────────────
    return ('other', float('inf'), 'decoder')


# ─────────────────────────────────────────────────────────────────────────────
# Parameter statistics
# ─────────────────────────────────────────────────────────────────────────────

def param_stats(tensor):
    t = tensor.float()
    with torch.no_grad():
        has_nan = bool(torch.isnan(t).any())
        has_inf = bool(torch.isinf(t).any())
        numel = t.numel()
        if numel == 0:
            return {"norm": 0.0, "mean": 0.0, "std": 0.0,
                    "max_abs": 0.0, "nan": has_nan, "inf": has_inf}
        return {
            "norm":    t.norm(2).item(),
            "mean":    t.mean().item(),
            "std":     t.std(unbiased=False).item(),
            "max_abs": t.abs().max().item(),
            "nan":     has_nan,
            "inf":     has_inf,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

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
            "file":             f.name,
            "epoch_num":        epoch_num,
            "ck_epoch":         ck.get("epoch", "?"),
            "loss":             ck.get("loss", float("nan")),
            "optimizer_steps":  ck.get("optimizer_steps_completed",
                                       ck.get("current_optimizer_step", "?")),
            "ema_updates":      ck.get("ema_updates", "?"),
            "state_dict":       ck.get("model_state_dict") or ck.get("state_dict") or {},
            "ema_state_dict":   ck.get("ema_model_state_dict") or {},
            "config":           ck.get("config"),
        }
        records.append(rec)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Weight statistics + deltas
# ─────────────────────────────────────────────────────────────────────────────

def compute_weight_stats(records):
    """
    Add per-parameter stats, live-weight deltas, and EMA-vs-live divergence.

    Memory note: we only retain float32 copies of the *previous* checkpoint's
    weights for the delta computation, then immediately release them.  The full
    state_dict tensors are kept as loaded (usually fp32 or bf16) for norm
    reporting; we do NOT clone the entire state_dict a second time.
    """
    prev_state = None   # dict[name -> float32 tensor] from previous epoch only

    for rec in records:
        sd      = rec["state_dict"]
        ema_sd  = rec["ema_state_dict"]
        stats   = {}
        any_nan = any_inf = False
        total_norm_sq = delta_norm_sq = ema_delta_norm_sq = 0.0
        n_params = 0

        with torch.no_grad():
            for name, tensor in sd.items():
                s = param_stats(tensor)
                stats[name] = s
                any_nan = any_nan or s["nan"]
                any_inf = any_inf or s["inf"]
                total_norm_sq += s["norm"] ** 2
                n_params      += tensor.numel()

                # Delta from previous live checkpoint
                if prev_state is not None and name in prev_state:
                    diff = (tensor.float() - prev_state[name]).norm(2).item()
                    stats[name]["delta"] = diff
                    delta_norm_sq += diff ** 2
                else:
                    stats[name]["delta"] = None

            # EMA vs live divergence  (only when EMA dict is present)
            ema_div_norm_sq = 0.0
            has_ema = bool(ema_sd)
            if has_ema:
                for name, tensor in sd.items():
                    ema_t = ema_sd.get(name)
                    if ema_t is not None:
                        ema_div = (tensor.float() - ema_t.float()).norm(2).item()
                        ema_div_norm_sq += ema_div ** 2

        rec["param_stats"]           = stats
        rec["total_weight_norm"]     = math.sqrt(total_norm_sq)
        rec["total_delta_norm"]      = math.sqrt(delta_norm_sq) if prev_state is not None else None
        rec["ema_divergence_norm"]   = math.sqrt(ema_div_norm_sq) if has_ema else None
        rec["any_nan"]               = any_nan
        rec["any_inf"]               = any_inf
        rec["n_params"]              = n_params

        # Delta velocity: ||Δw|| / optimizer_steps_in_epoch  (normalises for step count)
        steps_this_epoch = None
        if (prev_state is not None
                and isinstance(rec["optimizer_steps"], int)
                and isinstance(records[records.index(rec) - 1]["optimizer_steps"], int)):
            steps_this_epoch = (rec["optimizer_steps"]
                                - records[records.index(rec) - 1]["optimizer_steps"])
        rec["delta_velocity"] = (
            rec["total_delta_norm"] / steps_this_epoch
            if (rec["total_delta_norm"] is not None and steps_this_epoch and steps_this_epoch > 0)
            else None
        )

        # Build minimal float32 prev_state for next iteration — release old one
        prev_state = {k: v.float() for k, v in sd.items()}

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Rank-stability analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_rank_stability(records, top_n=10):
    """
    For each consecutive checkpoint pair, compute the Jaccard overlap between
    the top-N highest-delta parameter names.  Low overlap = different layers
    are destabilising each epoch (diffuse drift).  Consistently low overlap
    for the *same* layers = a persistent mover.

    Adds rec["top_delta_names"] and rec["rank_stability_jaccard"] to each record.
    Also returns a dict of param_name -> count of epochs in which that param
    appeared in the top-N.
    """
    persistent_counts = defaultdict(int)

    for rec in records:
        ps = rec.get("param_stats", {})
        deltas = [(name, s["delta"]) for name, s in ps.items()
                  if s.get("delta") is not None]
        deltas.sort(key=lambda x: x[1], reverse=True)
        top = [name for name, _ in deltas[:top_n]]
        rec["top_delta_names"] = top
        for name in top:
            persistent_counts[name] += 1

    # Jaccard between consecutive epochs
    for i in range(1, len(records)):
        prev_top = set(records[i - 1].get("top_delta_names", []))
        curr_top = set(records[i].get("top_delta_names", []))
        if prev_top or curr_top:
            jaccard = len(prev_top & curr_top) / len(prev_top | curr_top)
        else:
            jaccard = 1.0
        records[i]["rank_stability_jaccard"] = jaccard
    if records:
        records[0]["rank_stability_jaccard"] = None

    return records, persistent_counts


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables (checkpoint-level)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(records):
    print("\n" + "=" * 130)
    print(f"{'File':<38} {'Ep':>3} {'Loss':>10} {'Steps':>7} {'EMA Upd':>8} "
          f"{'Wt Norm':>10} {'Delta Norm':>11} {'Δ Velocity':>11} "
          f"{'EMA Div':>9} {'Jaccard':>8} {'NaN':>4} {'Inf':>4}")
    print("=" * 130)
    for r in records:
        loss_str    = (f"{r['loss']:.6f}"
                       if isinstance(r['loss'], float) and not math.isnan(r['loss'])
                       else str(r['loss']))
        delta_str   = f"{r['total_delta_norm']:.4f}"   if r['total_delta_norm']   is not None else "  --  "
        vel_str     = f"{r['delta_velocity']:.6f}"     if r['delta_velocity']     is not None else "  --  "
        ema_str     = f"{r['ema_divergence_norm']:.3f}" if r['ema_divergence_norm'] is not None else "  --  "
        jac_str     = f"{r['rank_stability_jaccard']:.2f}" if r['rank_stability_jaccard'] is not None else "  --"
        nan_flag    = "YES" if r["any_nan"] else "."
        inf_flag    = "YES" if r["any_inf"] else "."
        print(f"{r['file']:<38} {r['epoch_num']:>3} {loss_str:>10} "
              f"{str(r['optimizer_steps']):>7} {str(r['ema_updates']):>8} "
              f"{r['total_weight_norm']:>10.2f} {delta_str:>11} {vel_str:>11} "
              f"{ema_str:>9} {jac_str:>8} {nan_flag:>4} {inf_flag:>4}")
    print("=" * 130)


def print_key_layer_table(records):
    print("\n--- Key layer weight norms per epoch ---")
    epoch_labels = [f"Ep{r['epoch_num']:02d}" for r in records]
    print(f"{'Layer':<60} " + "  ".join(f"{e:>8}" for e in epoch_labels))
    print("-" * (60 + 10 * len(records)))
    all_names = list(records[0]["param_stats"].keys()) if records else []
    matched   = [n for n in all_names if any(k in n for k in KEY_LAYERS)]
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
    matched   = [n for n in all_names if any(k in n for k in KEY_LAYERS)]
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
    print("\n--- Top-10 largest weight changes at each epoch transition ---")
    for r in records:
        if r["total_delta_norm"] is None:
            continue
        deltas = [(name, s["delta"]) for name, s in r["param_stats"].items()
                  if s.get("delta") is not None]
        deltas.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  → {r['file']}  (total delta norm: {r['total_delta_norm']:.4f})")
        for name, d in deltas[:10]:
            print(f"    {d:>10.5f}  {name}")


def print_persistent_movers(persistent_counts, n_epochs, top_n=10):
    """Print parameters that appear in the top-delta list most consistently."""
    print("\n--- Persistent high-delta parameters (potential non-converging layers) ---")
    print(f"  (appearing in top-{top_n} largest-delta list across {n_epochs} epoch transitions)\n")
    ranked = sorted(persistent_counts.items(), key=lambda x: x[1], reverse=True)
    for name, count in ranked[:20]:
        pct = 100.0 * count / max(n_epochs, 1)
        bar = "█" * count + "░" * (n_epochs - count)
        flag = "  ← PERSISTENT MOVER" if pct >= 60 else ""
        print(f"  {count:>2}/{n_epochs} [{bar}]  {pct:>5.1f}%  {name}{flag}")


def flag_regression_epoch(records):
    """
    Find epoch with largest delta norm jump that also coincides with a loss
    increase.  Pure weight jumps on improving loss are not flagged as regression.
    """
    candidates = [
        (r["total_delta_norm"], r)
        for r in records
        if r["total_delta_norm"] is not None
    ]
    if not candidates:
        return
    candidates.sort(reverse=True)
    top = candidates[:3]
    print("\n--- Largest overall weight changes (top 3 candidates for regression onset) ---")
    for delta, r in top:
        loss_val = r['loss']
        loss_str = f"{loss_val:.6f}" if isinstance(loss_val, float) else str(loss_val)
        # Cross-reference with loss trend
        idx = next((i for i, rec in enumerate(records) if rec is r), None)
        if idx is not None and idx > 0:
            prev_loss = records[idx - 1]['loss']
            if (isinstance(loss_val, float) and isinstance(prev_loss, float)
                    and not math.isnan(loss_val) and not math.isnan(prev_loss)):
                loss_note = (f"  loss went UP +{loss_val - prev_loss:+.5f}  ← likely regression onset"
                             if loss_val > prev_loss
                             else f"  loss went DOWN {loss_val - prev_loss:+.5f}  (weight jump on improving run)")
            else:
                loss_note = ""
        else:
            loss_note = ""
        print(f"  Epoch {r['epoch_num']:>3}  ({r['file']})  delta norm = {delta:.4f}  loss = {loss_str}{loss_note}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Gradual degradation detector
# ─────────────────────────────────────────────────────────────────────────────

def _get_cfg_from_records(records):
    """Extract config from the latest checkpoint record, or fall back to TrainingConfig."""
    cfg = None
    if records:
        try:
            last = max(records, key=lambda r: r["epoch_num"])
            cfg = last.get("config")
        except Exception:
            pass
    if cfg is None:
        try:
            from kokoro.training.config import TrainingConfig
            cfg = TrainingConfig()
        except Exception:
            pass
    return cfg


def _spec_augment_display_epoch(cfg):
    """Return 1-indexed display epoch where SpecAugment activates, or None.

    The training loop is 0-indexed (epoch=0 → Ep01), so the display epoch
    is ``spec_augment_start_epoch + 1``.
    """
    if cfg is None:
        return None
    if not getattr(cfg, 'use_spec_augment', False):
        return None
    start = getattr(cfg, 'spec_augment_start_epoch', None)
    if start is None:
        return None
    return start + 1


def _compute_sa_window(vm_vals, sa_epoch):
    """Compute effective SpecAugment adaptation window from val_mel data.

    Extends beyond the default 5 if val_mel keeps rising continuously
    from SA onset — the adaptation tail can exceed 5 epochs.
    """
    if sa_epoch is None or not vm_vals:
        return 5
    start_idx = sa_epoch - 1  # 0-based
    if start_idx <= 0 or start_idx >= len(vm_vals):
        return 5
    window = 0
    for i in range(start_idx, len(vm_vals)):
        if vm_vals[i] > vm_vals[i - 1]:
            window += 1
        else:
            break
    return max(5, window)


def _is_spec_augment_transient(epoch_1indexed, sa_epoch, window=5, vm_vals=None):
    """True if *epoch_1indexed* falls within the SpecAugment adaptation window.

    If *vm_vals* (list of val_mel values) is provided, the window is extended
    dynamically as long as val_mel keeps rising continuously from SA onset.
    """
    if sa_epoch is None:
        return False
    if vm_vals is not None:
        window = _compute_sa_window(vm_vals, sa_epoch)
    return sa_epoch <= epoch_1indexed < sa_epoch + window


def _linear_slope(ys):
    """Least-squares slope over evenly-spaced indices.  Returns (slope, r_squared)."""
    n = len(ys)
    if n < 2:
        return 0.0, 0.0
    xs   = list(range(n))
    xbar = (n - 1) / 2.0
    ybar = sum(ys) / n
    ssxx = sum((x - xbar) ** 2 for x in xs)
    ssxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    if ssxx == 0:
        return 0.0, 0.0
    slope = ssxy / ssxx
    ss_res = sum((y - (ybar + slope * (x - xbar))) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


def print_gradual_degradation_report(records):
    """
    Detects slow, sustained drift across checkpoints — the pattern that
    spike-based checks miss entirely.

    Signals computed:
      1. Total-delta-norm slope  — is the model moving faster over time?
      2. Delta velocity slope    — acceleration after normalising for step count
      3. EMA divergence trend    — is live model drifting away from its own EMA?
      4. Loss slope              — simple linear fit across all epochs
      5. Weight-norm drift       — sustained growth/shrinkage of total weight norm
      6. "Best epoch age"        — how many epochs ago was the best checkpoint?
    """
    print("\n" + "=" * 90)
    print("CHECKPOINT — Gradual Degradation Report")
    print("=" * 90)

    n = len(records)
    if n < 3:
        print("  Need ≥ 3 checkpoints for trend analysis.")
        return

    # ── 1. Loss trend ────────────────────────────────────────────────────────
    losses = [r['loss'] for r in records
              if isinstance(r['loss'], float) and not math.isnan(r['loss'])]
    if len(losses) >= 3:
        best_loss  = min(losses)
        best_idx   = losses.index(best_loss)
        best_epoch = records[best_idx]['epoch_num']
        last_epoch = records[-1]['epoch_num']
        age        = last_epoch - best_epoch

        # Full-series slope (used for overall direction reporting)
        slope_all, r2_all = _linear_slope(losses)

        # Post-best-epoch slope — the meaningful regression signal.
        # A negative slope on the full series can mask a clear upward trend
        # after the best epoch if early epochs pull the fit down (as seen when
        # epoch 1 loss >> epoch 2+).
        post_best = losses[best_idx:]
        if len(post_best) >= 3:
            slope_post, r2_post = _linear_slope(post_best)
        elif len(post_best) == 2:
            # Only 1 epoch post-best: 2-point fit always gives R²=1.0, which
            # would spuriously trigger the SUSTAINED REGRESSION flag.  Compute
            # slope but zero R² so no flag fires; the raw slope is still shown.
            slope_post, _ = _linear_slope(post_best)
            r2_post = 0.0
        else:
            slope_post, r2_post = slope_all, r2_all

        slope_flag = ""
        if slope_post > 0 and r2_post > 0.5:
            slope_flag = "  ← TRAIN LOSS rising post-best  (R²={:.2f})  [see TB val_mel for true regression signal]".format(r2_post)
        elif slope_post > 0 and r2_post > 0.25:
            slope_flag = "  ← train loss upward trend post-best  (R²={:.2f})".format(r2_post)

        print(f"  Loss trend (all)  slope={slope_all:+.6f}/epoch  R²={r2_all:.3f}  (train loss)")
        print(f"  Loss trend (post-best Ep{best_epoch:02d})  "
              f"slope={slope_post:+.6f}/epoch  R²={r2_post:.3f}{slope_flag}")
        if age >= 2:
            print(f"  Best epoch        Ep{best_epoch:02d}  ({age} epochs ago, train loss)"
                  f"  ← train loss has not improved; check TB val_mel for true best")
        else:
            print(f"  Best epoch        Ep{best_epoch:02d}  (most recent best, train loss)")
    else:
        print("  Loss trend        insufficient float loss values")

    # ── 2. Delta norm trend  (is the model moving faster or slower?) ─────────
    # Normalise each delta by its epoch gap so that multi-epoch spans (e.g.
    # Ep4→Ep6 when Ep5 is missing) don't inflate the acceleration signal.
    delta_records = [r for r in records if r['total_delta_norm'] is not None]
    if len(delta_records) >= 3:
        delta_per_ep = []
        has_gap = False
        for dr in delta_records:
            idx = records.index(dr)
            epoch_gap = dr['epoch_num'] - records[idx - 1]['epoch_num']
            if epoch_gap > 1:
                has_gap = True
            delta_per_ep.append(dr['total_delta_norm'] / max(epoch_gap, 1))
        slope, r2  = _linear_slope(delta_per_ep)
        accel_flag = ""
        if slope > 0 and r2 > 0.4:
            accel_flag = "  ← weight drift ACCELERATING  (R²={:.2f})".format(r2)
        elif slope < 0 and r2 > 0.4:
            accel_flag = "  ← weight drift decelerating (converging)  (R²={:.2f})".format(r2)
        gap_note = "  [epoch-gap normalised]" if has_gap else ""
        print(f"  Δ-norm trend      slope={slope:+.4f}/epoch  R²={r2:.3f}{accel_flag}{gap_note}")
    else:
        print("  Δ-norm trend      insufficient data")

    # ── 3. Delta velocity trend  (drift per optimizer step) ─────────────────
    vels = [r['delta_velocity'] for r in records if r['delta_velocity'] is not None]
    if len(vels) >= 3:
        slope, r2  = _linear_slope(vels)
        vel_flag   = ""
        if slope > 0 and r2 > 0.4:
            vel_flag = "  ← drift velocity INCREASING (not explained by step count)"
        print(f"  Δ-velocity trend  slope={slope:+.8f}/step  R²={r2:.3f}{vel_flag}")

    # ── 4. EMA divergence trend ─────────────────────────────────────────────
    ema_divs = [(r['epoch_num'], r['ema_divergence_norm'])
                for r in records if r['ema_divergence_norm'] is not None]
    if len(ema_divs) >= 3:
        ema_vals  = [v for _, v in ema_divs]
        slope, r2 = _linear_slope(ema_vals)
        ema_flag  = ""
        if slope > 0 and r2 > 0.5:
            ema_flag = ("  ← GROWING EMA DIVERGENCE  (R²={:.2f})\n"
                        "              live model oscillating; EMA lags by "
                        "{:.3f} norm units per epoch".format(r2, slope))
        elif slope < 0 and r2 > 0.4:
            ema_flag = "  ← EMA converging toward live model (healthy)"
        print(f"  EMA div trend     slope={slope:+.4f}/epoch  R²={r2:.3f}{ema_flag}")
        # Print raw EMA divergence per epoch for reference
        print("  EMA divergence    " + "  ".join(
            f"Ep{ep:02d}:{v:.3f}" for ep, v in ema_divs))
    else:
        print("  EMA divergence    insufficient data or no EMA state dict in checkpoints")

    # ── 5. Weight-norm drift ─────────────────────────────────────────────────
    wnorms = [r['total_weight_norm'] for r in records]
    if len(wnorms) >= 3:
        slope, r2  = _linear_slope(wnorms)
        wn_flag    = ""
        if abs(slope) > 0.5 and r2 > 0.5:
            direction = "GROWING" if slope > 0 else "SHRINKING"
            wn_flag   = f"  ← weight norm {direction}  (R²={r2:.2f})"
        print(f"  Weight-norm trend slope={slope:+.4f}/epoch  R²={r2:.3f}{wn_flag}")

    # ── 6. Per-epoch summary strip ───────────────────────────────────────────
    print()
    print(f"  {'Ep':>3}  {'loss':>10}  {'Δnorm':>9}  {'Δvel×1e4':>9}  "
          f"{'EMA div':>8}  {'wt norm':>9}  {'Jaccard':>8}")
    print("  " + "-" * 70)
    prev_loss = None
    for r in records:
        loss_val = r['loss']
        loss_str = f"{loss_val:.6f}" if isinstance(loss_val, float) and not math.isnan(loss_val) else "       ?"
        loss_arrow = ""
        if prev_loss is not None and isinstance(loss_val, float) and not math.isnan(loss_val):
            loss_arrow = " ▲" if loss_val > prev_loss else " ▼"
        delta_str = f"{r['total_delta_norm']:>9.4f}" if r['total_delta_norm'] is not None else "       --"
        vel_str   = f"{r['delta_velocity']*1e4:>9.4f}" if r['delta_velocity'] is not None else "       --"
        ema_str   = f"{r['ema_divergence_norm']:>8.3f}"  if r['ema_divergence_norm'] is not None else "      --"
        jac_str   = f"{r['rank_stability_jaccard']:>8.2f}" if r['rank_stability_jaccard'] is not None else "      --"
        print(f"  {r['epoch_num']:>3}  {loss_str}{loss_arrow:<2}  {delta_str}  {vel_str}  "
              f"{ema_str}  {r['total_weight_norm']:>9.2f}  {jac_str}")
        if isinstance(loss_val, float) and not math.isnan(loss_val):
            prev_loss = loss_val

    print()


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tb(log_dir: Path):
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


def _pct(series, p):
    """Return the p-th percentile (0..100) of values in series.  Safe on empty input."""
    if not series:
        return 0.0
    vals = sorted(v for _, v in series)
    idx  = max(0, min(len(vals) - 1, int(len(vals) * p / 100)))
    return vals[idx]


def _series_stats(series):
    if not series:
        return None
    vals = [v for _, v in series]
    return {
        "n":          len(vals),
        "first":      vals[0],
        "last":       vals[-1],
        "mean":       statistics.mean(vals),
        "min":        min(vals),
        "max":        max(vals),
        "step_first": series[0][0],
        "step_last":  series[-1][0],
        "trend":      "UP ▲" if vals[-1] > vals[0] else "DOWN ▼",
        "delta":      vals[-1] - vals[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW: TB gradual degradation — per-loss-component trend table
# ─────────────────────────────────────────────────────────────────────────────

def tb_print_per_loss_trend(ea, cfg=None):
    """
    For each tracked loss component, fit a linear slope across epoch-level
    validation values.  Surfaces components that are quietly climbing while the
    aggregate looks stable — the primary cause of gradual degradation.
    """
    sa_epoch = _spec_augment_display_epoch(cfg)
    print("\n" + "=" * 90)
    print("TENSORBOARD — Per-Loss Component Trend (gradual degradation detector)")
    print("=" * 90)

    loss_tags = [
        ("mel",       "loss/val_mel_epoch"),
        ("stop",      "loss/val_stop_epoch"),
        ("duration",  "loss/val_duration_epoch"),
        ("pitch",     "metrics/val_f0_rmse"),
        ("energy",    "loss/val_energy_epoch" ),
        ("total",     "loss/val_total_epoch"),
        ("spec_conv", "metrics/val_spectral_convergence"),
    ]

    header_printed = False
    for label, tag in loss_tags:
        series = _get(ea, tag)
        if len(series) < 3:
            continue
        if not header_printed:
            print(f"  {'Component':<12} {'N ep':>4}  {'first':>9}  {'last':>9}  "
                  f"{'slope/ep':>10}  {'R²':>6}  {'best ep':>7}  {'best age':>8}  verdict")
            print("  " + "-" * 82)
            header_printed = True
        vals        = [v for _, v in series]
        slope, r2   = _linear_slope(vals)
        best_val    = min(vals)
        best_ep_idx = vals.index(best_val) + 1
        best_age    = len(vals) - best_ep_idx  # epochs since best

        # If SpecAugment is active and the best epoch predates the onset,
        # recompute best age AND best epoch from the post-augment segment
        # so the displayed pair is consistent.
        display_best_ep = best_ep_idx
        display_best_age = best_age
        if sa_epoch is not None and best_ep_idx < sa_epoch and len(vals) >= sa_epoch:
            sa_vals = vals[sa_epoch - 1:]  # post-augment segment
            if sa_vals:
                sa_best_idx = sa_vals.index(min(sa_vals))
                display_best_age = len(sa_vals) - sa_best_idx - 1
                display_best_ep = sa_epoch + sa_best_idx

        # SA adaptation tail: post-SA segment still rising even if overall slope is negative
        if (sa_epoch is not None and best_ep_idx < sa_epoch
                and len(vals) >= sa_epoch
                and vals[-1] > vals[sa_epoch - 1]):
            verdict = f"SpecAugment adapting (pre-SA best Ep{best_ep_idx:02d})"
        elif slope > 0 and r2 > 0.6:
            if sa_epoch is not None and best_ep_idx < sa_epoch:
                verdict = f"SpecAugment adapting (pre-SA best Ep{best_ep_idx:02d})"
            else:
                verdict = "SUSTAINED REGRESSION ▲"
        elif slope > 0 and r2 > 0.3:
            if sa_epoch is not None and best_ep_idx < sa_epoch:
                verdict = f"SpecAugment adapting (pre-SA best Ep{best_ep_idx:02d})"
            else:
                verdict = "weak upward drift ▲"
        elif slope < 0 and r2 > 0.4:
            verdict = "improving ▼"
        elif display_best_age >= 2:
            verdict = f"plateaued (best {display_best_age} ep ago)"
        else:
            verdict = "stable"

        print(f"  {label:<12} {len(vals):>4}  {vals[0]:>9.5f}  {vals[-1]:>9.5f}  "
              f"{slope:>+10.6f}  {r2:>6.3f}  {'Ep'+str(display_best_ep):>7}  "
              f"{display_best_age:>8}  {verdict}")

    if not header_printed:
        print("  No epoch-level loss tags with ≥ 3 data points found.")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Existing TB print functions (unchanged externally, bugs fixed internally)
# ─────────────────────────────────────────────────────────────────────────────

def tb_print_step_loss_summary(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Step-level Training Loss Summary")
    print("=" * 90)
    tags = ["loss/total", "loss/mel", "loss/stop", "loss/duration", "loss/pitch", "loss/energy"]
    print(f"  {'Tag':<28} {'N':>5}  {'Steps':>10}  {'First':>9}  {'Last':>9}  "
          f"{'Δ':>9}  {'Trend':>7}  {'Mean':>9}  {'Min':>9}  {'Max':>9}")
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


def tb_print_val_mel_series(ea, cfg=None):
    vm = _get(ea, "loss/val_mel_epoch")
    if not vm:
        return
    sa_epoch = _spec_augment_display_epoch(cfg)
    vm_vals = [v for _, v in vm]
    print("\n" + "=" * 90)
    print("TENSORBOARD — Val Mel Epoch Series")
    print("=" * 90)
    prev = None
    for i, (s, v) in enumerate(vm):
        ep = i + 1
        if prev is None:
            flag = ""
        elif v > prev:
            if _is_spec_augment_transient(ep, sa_epoch, vm_vals=vm_vals):
                flag = f"  ▲ +{v - prev:.5f}  (SpecAugment adaptation)"
            else:
                flag = f"  ▲ +{v - prev:.5f}  ← REGRESSION"
        else:
            flag = f"  ▼ {v - prev:+.5f}"
        print(f"  Ep{i+1:02d}  step={s:>5}  val_mel={v:.5f}{flag}")
        prev = v
    if len(vm) >= 2:
        vals     = [v for _, v in vm]
        best     = min(vals)
        best_ep  = vals.index(best) + 1
        slope, r2 = _linear_slope(vals)
        print(f"\n  best={best:.5f} at Ep{best_ep:02d}  "
              f"total Δ={vals[-1] - vals[0]:+.5f}  last={vals[-1]:.5f}  "
              f"linear slope={slope:+.6f}/ep  R²={r2:.3f}")


def tb_print_epoch_table(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Epoch-level Train vs Val Metrics")
    print("=" * 90)
    epoch_tags = [
        ("mel",       "loss/train_mel_epoch",      "loss/val_mel_epoch"),
        ("total",     "loss/train_total_epoch",    "loss/val_total_epoch"),
        ("stop",      "loss/train_stop_epoch",     "loss/val_stop_epoch"),
        ("duration",  "loss/train_duration_epoch", "loss/val_duration_epoch"),
        ("spec_conv", "metrics/train_spectral_convergence", "metrics/val_spectral_convergence"),
        ("f0_rmse",   None,                        "metrics/val_f0_rmse"),
    ]
    val_mel_series = _get(ea, "loss/val_mel_epoch")
    if not val_mel_series:
        print("  No epoch-level data available yet.")
        return
    epoch_steps = [s for s, _ in val_mel_series]
    n_epochs    = len(epoch_steps)

    # FIX: header was built but the label row was never printed
    header = (f"  {'Metric':<12}  "
              + "  ".join(f"  Ep{i+1:02d} (s{s:04d})" for i, s in enumerate(epoch_steps)))
    print(header)
    print("  " + "-" * (14 + 18 * n_epochs))

    for label, train_tag, val_tag in epoch_tags:
        train_vals = {s: v for s, v in (_get(ea, train_tag) if train_tag else [])}
        val_vals   = {s: v for s, v in _get(ea, val_tag)} if val_tag else {}

        if train_tag:
            row_t = f"  {label + ' train':<14}  "
            for s in epoch_steps:
                v = train_vals.get(s)
                row_t += f"  {'?':>14}  " if v is None else f"  {v:>12.5f}  "
            print(row_t)

        if val_tag:
            row_v      = f"  {label + ' val':<16}  "
            vals_list  = [val_vals.get(s) for s in epoch_steps]
            for i, (s, v) in enumerate(zip(epoch_steps, vals_list)):
                flag = ""
                if i > 0:
                    prev = vals_list[i - 1]
                    if v is not None and prev is not None and v > prev:
                        flag = " ▲"
                row_v += f"  {'?':>14}  " if v is None else f"  {v:>10.5f}{flag:2s}  "
            print(row_v)
        print()


def tb_print_stop_token_analysis(ea, cfg=None):
    sa_epoch = _spec_augment_display_epoch(cfg)
    _vm_raw = _get(ea, "loss/val_mel_epoch")
    _vm_vals = [v for _, v in _vm_raw] if _vm_raw else None
    print("\n" + "=" * 90)
    print("TENSORBOARD — Stop Token Analysis")
    print("=" * 90)

    # ── Stop head isolation parameters ──────────────────────────────────────
    print("  Stop head isolation parameters (from loaded config / defaults):")
    # Read from last available checkpoint
    stop_head = _get(ea, "stats/lr_stop_head")
    dec_series = _get(ea, "stats/lr_decoder")
    dec_lr_curr = dec_series[-1][1] if dec_series else None
    if stop_head:
        stop_lr_curr = stop_head[-1][1]
        mult_actual  = (stop_lr_curr / dec_lr_curr) if dec_lr_curr and dec_lr_curr > 0 else None
        print(f"    stats/lr_stop_head : {stop_lr_curr:.8f}  "
              + (f"({mult_actual:.4f}× decoder LR)  " if mult_actual is not None else "")
              + "[dedicated stop head param group active]")
    else:
        print("    stats/lr_stop_head : not found — stop head may share decoder LR group")
    ffn_series = _get(ea, "stats/lr_decoder_ffn")
    if ffn_series:
        ffn_lr_curr = ffn_series[-1][1]
        ffn_mult    = (ffn_lr_curr / dec_lr_curr) if dec_lr_curr and dec_lr_curr > 0 else None
        print(f"    stats/lr_decoder_ffn: {ffn_lr_curr:.8f}  "
              + (f"({ffn_mult:.4f}× decoder LR)  " if ffn_mult is not None else "")
              + "[dedicated decoder FFN param group active]")
    print("    gradient isolation : decoder_outputs detached before stop head (loss cannot")
    print("                         back-propagate through mel decoder from BCE stop loss)")
    print()
    series = _get(ea, "loss/stop")
    if not series:
        print("  No step-level stop data.")
    else:
        vals         = [v for _, v in series]
        p50          = _pct(series, 50)
        p90          = _pct(series, 90)
        p99          = _pct(series, 99)
        burst_thresh = p50 * 2
        bursts       = [(s, v) for s, v in series if v > burst_thresh]
        late_bursts  = [(s, v) for s, v in bursts if s > (series[-1][0] * 0.5)]
        print(f"  Step-level loss/stop ({len(series)} points, steps {series[0][0]}–{series[-1][0]})")
        print(f"    first={vals[0]:.5f}  last={vals[-1]:.5f}  Δ={vals[-1]-vals[0]:+.5f}  "
              f"mean={statistics.mean(vals):.5f}")
        print(f"    min={min(vals):.5f}  max={max(vals):.5f}")
        print(f"    p50={p50:.5f}  p90={p90:.5f}  p99={p99:.5f}")
        print(f"    bursts > 2×median ({burst_thresh:.4f}):  total={len(bursts)}  "
              f"in 2nd half={len(late_bursts)}")
        if bursts:
            print(f"    burst steps: {[s for s, _ in bursts[:15]]}"
                  + (" ..." if len(bursts) > 15 else ""))
        if late_bursts:
            print(f"    ⚠  Late bursts (2nd half of run): {late_bursts[:8]}")
        else:
            print(f"    ✓  No bursts in 2nd half of run — stop loss stabilizing.")

    for tag, label in [("loss/train_stop_epoch", "train"), ("loss/val_stop_epoch", "val")]:
        ep = _get(ea, tag)
        if ep:
            print(f"\n  Epoch-level stop ({label}):")
            for i, (s, v) in enumerate(ep):
                prev_v = ep[i - 1][1] if i > 0 else None
                if prev_v is not None and v > prev_v:
                    if _is_spec_augment_transient(i + 1, sa_epoch, vm_vals=_vm_vals):
                        flag = " ▲ (SpecAugment)"
                    else:
                        flag = " ▲ REGRESSION"
                else:
                    flag = ""
                print(f"    Ep{i+1:02d}  step={s:>5}  {v:.5f}{flag}")


def tb_print_gradient_analysis(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Gradient Health")
    print("=" * 90)
    gn  = _get(ea, "stats/grad_norm")
    gnc = _get(ea, "stats/grad_norm_clipped")

    if not gn:
        print("  No grad_norm data.")
        return

    raw_vals     = [v for _, v in gn]
    clipped_vals = [v for _, v in gnc] if gnc else []

    print(f"  grad_norm  ({len(gn)} steps, {gn[0][0]}–{gn[-1][0]})")
    print(f"    first={raw_vals[0]:.4f}  last={raw_vals[-1]:.4f}  "
          f"mean={statistics.mean(raw_vals):.4f}  max={max(raw_vals):.4f}")

    for thresh in [5.0, 10.0, 20.0]:
        spikes      = [(s, v) for s, v in gn if v > thresh]
        step_range  = gn[-1][0] - gn[0][0]
        early_cut   = gn[0][0] + step_range * 0.10
        late_spikes = [(s, v) for s, v in spikes if s > early_cut]
        print(f"    spikes > {thresh:4.1f}: total={len(spikes):4d}  "
              f"after first 10% of steps={len(late_spikes):4d}"
              + (f"  e.g. {late_spikes[:4]}" if late_spikes else ""))

    if clipped_vals:
        max_clip = max(clipped_vals)
        at_cap   = sum(1 for v in clipped_vals if abs(v - max_clip) < 1e-4)
        pct      = 100.0 * at_cap / len(clipped_vals)
        status   = ("✓ HEALTHY" if pct < 20
                    else ("⚠ ELEVATED" if pct < 40 else "✗ SATURATED"))
        print(f"\n  Clipping saturation: {at_cap}/{len(clipped_vals)} steps at cap "
              f"({pct:.1f}%)  cap={max_clip:.4f}  → {status}")

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
                    print(f"    Ep{i+1:02d} (steps {prev_b+1} –{b:>4}): "
                          f"{w_sat:>4}/{len(window):<5} = {w_pct:5.1f}%{flag}")
                prev_b = b
            rest = [(s, v) for s, v in gnc if s > prev_b]
            if rest:
                r_cap = max(v for _, v in rest)
                r_sat = sum(1 for _, v in rest if abs(v - r_cap) < 1e-4)
                r_pct = 100.0 * r_sat / len(rest)
                flag  = "" if r_pct < 20 else (" ⚠" if r_pct < 40 else " ✗")
                print(f"    Ep?  (steps {prev_b+1} –{rest[-1][0]:>4}, in progress): "
                      f"{r_sat:>4}/{len(rest):<5} = {r_pct:5.1f}%{flag}")


def tb_print_lr_trajectory(ea):
    print("\n" + "=" * 90)
    print("TENSORBOARD — Learning Rate Trajectory")
    print("=" * 90)
    groups = [
        ("stats/lr_decoder",     "decoder"),
        ("stats/lr_decoder_ffn", "decoder_ffn"),
        ("stats/lr_encoder",     "encoder"),
        ("stats/lr_stop_head",   "stop_head"),
    ]
    for tag, label in groups:
        series = _get(ea, tag)
        if not series:
            continue
        vals = [v for _, v in series]
        peak_val = max(vals)
        last_val = vals[-1]
        # If the peak is significantly above the last value, it indicates a mid-run LR
        # group change (e.g. multiplier was lowered and training resumed from a checkpoint),
        # causing old TB events to show a higher historical max.
        if peak_val > last_val * 1.1:
            max_note = f"  ⚠ historical max from pre-resume run (lr group changed mid-training)"
        else:
            max_note = ""
        print(f"  {label}: first={vals[0]:.7f}  last={last_val:.7f}  "
              f"max={peak_val:.7f}  trend={'UP ▲' if last_val > vals[0] else 'DOWN ▼'}{max_note}")
        n         = len(series)
        step_size = max(1, n // 8)
        samples   = series[::step_size] + [series[-1]]
        seen      = set()
        for s, v in samples:
            if s not in seen:
                print(f"    step={s:>5}  {v:.8f}")
                seen.add(s)
    dec = _get(ea, "stats/lr_decoder")
    if dec:
        max_lr   = max(v for _, v in dec)
        curr_lr  = dec[-1][1]
        tail     = [v for _, v in dec[-10:]]
        still_rising = len(tail) >= 2 and tail[-1] > tail[0]
        if still_rising and curr_lr >= max_lr * 0.99:
            print(f"\n  Phase: WARMUP (still rising)")
        elif curr_lr < max_lr * 0.99:
            pct = 100.0 * curr_lr / max_lr
            print(f"\n  Phase: DECAY  (at {pct:.1f}% of peak LR {max_lr:.7f})")
        else:
            print(f"\n  Phase: AT PEAK  (lr={curr_lr:.7f})")


def tb_print_mel_stop_window_correlation(ea):
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

    max_step  = mel[-1][0]
    lr_max    = max(v for _, v in lr) if lr else 1.0
    lr_lookup = {s: v for s, v in lr}

    prev_mel_m = prev_stop_m = None
    first_step = mel[0][0]
    w_start    = (first_step // 200) * 200
    w_size     = 200
    # Pre-build stop lookup for safe nearest-sample access
    stop_lookup = {s: v for s, v in stop} if stop else {}

    while w_start <= max_step:
        w_end    = w_start + w_size
        seg_mel  = [v for s, v in mel  if w_start <= s < w_end]
        seg_stop = [v for s, v in stop if w_start <= s < w_end] if stop else []
        mid      = w_start + w_size // 2
        lr_here  = min(lr_lookup.items(), key=lambda x: abs(x[0] - mid))[1] if lr_lookup else 0
        lr_pct   = 100.0 * lr_here / lr_max if lr_max else 0

        if seg_mel:
            mm    = statistics.mean(seg_mel)
            sm    = statistics.mean(seg_stop) if seg_stop else None
            dmel  = mm - prev_mel_m  if prev_mel_m is not None else None
            dstop = (sm - prev_stop_m) if (prev_stop_m is not None and sm is not None) else None

            mel_arr  = ("▲" if dmel  and dmel  > 0 else "▼") if dmel  is not None else " "
            stop_arr = ("▲" if dstop and dstop > 0 else "▼") if dstop is not None else " "

            comove = ""
            if dmel is not None and dstop is not None:
                if dmel > 0 and dstop > 0:
                    comove = "both↑ (LR pressure)"
                elif dmel < 0 and dstop < 0:
                    comove = "both↓ (improving)"
                elif dstop > 0 and dmel <= 0:
                    comove = "stop↑ only (stop source)"
                elif dmel > 0 and dstop <= 0:
                    comove = "mel↑ only"

            dmel_str  = f"{dmel:+.4f}"  if dmel  is not None else "      "
            dstop_str = f"{dstop:+.4f}" if dstop is not None else "      "
            sm_str    = f"{sm:.5f}"     if sm    is not None else "       ?"

            print(f"  {w_start:>5}–{w_end:<5}  {mm:>9.5f} {mel_arr} {dmel_str}  "
                  f"{sm_str} {stop_arr} {dstop_str}  {lr_pct:>5.1f}%  {comove}")

            prev_mel_m  = mm
            prev_stop_m = sm
        w_start = w_end


def _collect_diagnostics(ea):
    """
    Gather all raw signals into a dict of structured findings.
    Cached on the EventAccumulator instance to avoid re-reading TB on each call.
    BUG FIX: cache store was after `return d` — moved to before it.
    """
    cache_key = '_diag_cache'
    try:
        if ea is not None and hasattr(ea, cache_key):
            return getattr(ea, cache_key)
    except Exception:
        pass

    d   = {}
    gn        = _get(ea, "stats/grad_norm")
    gnc       = _get(ea, "stats/grad_norm_clipped")
    vm        = _get(ea, "loss/val_mel_epoch")
    tm        = _get(ea, "loss/train_mel_epoch")
    dec       = _get(ea, "stats/lr_decoder")
    stop_step = _get(ea, "loss/stop")
    stop_ep   = _get(ea, "loss/val_stop_epoch")

    # ── grad spikes ──────────────────────────────────────────────────────────
    if gn:
        step_range   = gn[-1][0] - gn[0][0]
        early_cutoff = gn[0][0] + step_range * 0.10
        late         = [(s, v) for s, v in gn if s > early_cutoff and v > 5.0]
        late_gt10    = [(s, v) for s, v in gn if s > early_cutoff and v > 10.0]
        late_gt20    = [(s, v) for s, v in gn if s > early_cutoff and v > 20.0]
        late_steps   = sorted(s for s, _ in late)
        clusters     = 0
        i = 0
        while i < len(late_steps):
            j = i + 1
            while j < len(late_steps) and late_steps[j] - late_steps[j - 1] <= 5:
                j += 1
            if j - i > 1:
                clusters += 1
            i = j
        d["late_spikes"]      = late
        d["late_spikes_gt10"] = late_gt10
        d["late_spikes_gt20"] = late_gt20
        d["spike_max"]        = max((v for _, v in late), default=0.0)
        d["spike_clusters"]   = clusters
        d["early_cutoff"]     = early_cutoff
    else:
        d["late_spikes"] = d["late_spikes_gt10"] = d["late_spikes_gt20"] = []
        d["spike_max"]   = 0.0
        d["spike_clusters"] = 0

    # ── clipping saturation ──────────────────────────────────────────────────
    if gnc:
        max_clip = max(v for _, v in gnc)
        at_cap   = sum(1 for _, v in gnc if abs(v - max_clip) < 1e-4)
        d["clip_cap"]         = max_clip
        d["clip_total_steps"] = len(gnc)
        d["clip_sat_pct"]     = 100.0 * at_cap / len(gnc)
        boundaries = [s for s, _ in vm] if vm else []
        epoch_sat  = []
        prev_b     = 0
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
            "improving"  if len(epoch_sat) >= 2 and epoch_sat[-1] < epoch_sat[0] else
            ("worsening" if len(epoch_sat) >= 2 and epoch_sat[-1] > epoch_sat[0] else "stable")
        )
    else:
        d["clip_sat_pct"]       = 0.0
        d["clip_sat_per_epoch"] = []
        d["clip_sat_trend"]     = "unknown"

    # ── val mel regression ───────────────────────────────────────────────────
    if vm:
        vm_vals = [v for _, v in vm]
        d["val_mel_series"]      = vm_vals
        d["val_mel_regressed"]   = vm_vals[-1] > vm_vals[0] if len(vm_vals) >= 2 else False
        d["val_mel_regressions"] = [
            (i + 2, vm_vals[i + 1] - vm_vals[i])
            for i in range(len(vm_vals) - 1) if vm_vals[i + 1] > vm_vals[i]
        ]
        d["val_mel_slope"], d["val_mel_r2"] = _linear_slope(vm_vals)
    else:
        d["val_mel_series"]      = []
        d["val_mel_regressed"]   = False
        d["val_mel_regressions"] = []
        d["val_mel_slope"]       = 0.0
        d["val_mel_r2"]          = 0.0

    # ── val/train mel gap ────────────────────────────────────────────────────
    if vm and tm:
        val_dict   = {s: v for s, v in vm}
        train_dict = {s: v for s, v in tm}
        common     = sorted(set(val_dict) & set(train_dict))
        gaps       = [val_dict[s] - train_dict[s] for s in common]
        d["val_train_gap"]        = gaps[-1] if gaps else None
        d["val_train_gap_series"] = gaps
    else:
        d["val_train_gap"]        = None
        d["val_train_gap_series"] = []

    # ── stop token ───────────────────────────────────────────────────────────
    if stop_step:
        p50          = _pct(stop_step, 50)
        bursts       = [(s, v) for s, v in stop_step if v > p50 * 2]
        step_range   = stop_step[-1][0] - stop_step[0][0]
        late_cut     = stop_step[0][0] + step_range * 0.5
        d["stop_late_bursts"] = [(s, v) for s, v in bursts if s > late_cut]
        d["stop_max"]         = max(v for _, v in stop_step)
    else:
        d["stop_late_bursts"] = []
        d["stop_max"]         = 0.0

    if stop_ep and vm:
        stop_dict = {s: v for s, v in stop_ep}
        mel_dict  = {s: v for s, v in vm}
        common    = sorted(set(stop_dict) & set(mel_dict))
        d["stop_mel_ratio"] = stop_dict[common[-1]] / mel_dict[common[-1]] if common else None
    else:
        d["stop_mel_ratio"] = None

    # ── stop/spike co-occurrence ─────────────────────────────────────────────
    if stop_step and d["late_spikes"]:
        stop_lookup = {s: v for s, v in stop_step}
        stop_p75    = _pct(stop_step, 75)
        co_occur    = []
        for (ss, sv) in d["late_spikes"]:
            for delta in range(-3, 4):
                sv_stop = stop_lookup.get(ss + delta)
                if sv_stop is not None and sv_stop > stop_p75:
                    co_occur.append((ss, sv, sv_stop))
                    break
        d["spike_stop_cooccur"]     = co_occur
        d["spike_stop_cooccur_pct"] = (
            100.0 * len(co_occur) / len(d["late_spikes"]) if d["late_spikes"] else 0.0
        )
    else:
        d["spike_stop_cooccur"]     = []
        d["spike_stop_cooccur_pct"] = 0.0

    # ── LR phase ─────────────────────────────────────────────────────────────
    if dec:
        max_lr   = max(v for _, v in dec)
        curr_lr  = dec[-1][1]
        tail     = [v for _, v in dec[-10:]]
        d["lr_max"]          = max_lr
        d["lr_current"]      = curr_lr
        d["lr_still_rising"] = len(tail) >= 2 and tail[-1] > tail[0]
        d["lr_phase"]        = (
            "warmup" if d["lr_still_rising"] and curr_lr >= max_lr * 0.99 else
            ("decay" if curr_lr < max_lr * 0.99 else "peak")
        )
        if gn:
            peak_step = next((s for s, v in dec if abs(v - max_lr) < max_lr * 0.02), None)
            d["lr_peak_step"]       = peak_step
            d["spikes_during_ramp"] = [(s, v) for s, v in d["late_spikes"] if peak_step and s <= peak_step]
            d["spikes_post_peak"]   = [(s, v) for s, v in d["late_spikes"] if peak_step and s > peak_step]
        else:
            d["lr_peak_step"]       = None
            d["spikes_during_ramp"] = d["late_spikes"]
            d["spikes_post_peak"]   = []
    else:
        d["lr_phase"]           = "unknown"
        d["lr_peak_step"]       = None
        d["spikes_during_ramp"] = []
        d["spikes_post_peak"]   = []

    # ── Gradual drift via val_mel linear slope ───────────────────────────────
    d["val_mel_gradual_drift"] = (
        d["val_mel_slope"] > 0 and d["val_mel_r2"] > 0.3
        and not d["val_mel_regressed"]  # not already caught by spike logic
    )

    # BUG FIX: cache store was after `return d` — now placed before it
    try:
        if ea is not None:
            setattr(ea, cache_key, d)
    except Exception:
        pass

    return d


def tb_print_regression_flags(ea, cfg=None):
    sa_epoch = _spec_augment_display_epoch(cfg)
    print("\n" + "=" * 90)
    print("TENSORBOARD — Regression Flag Summary")
    print("=" * 90)
    flags = []

    d = _collect_diagnostics(ea)

    # 1. Val mel overall regression
    vm = d["val_mel_series"]
    if len(vm) >= 2:
        if d["val_mel_regressed"]:
            flags.append(("FAIL", f"val_mel is HIGHER at last epoch ({vm[-1]:.5f}) vs first ({vm[0]:.5f})"))
        else:
            flags.append(("PASS", f"val_mel decreasing overall:  {vm[0]:.5f} → {vm[-1]:.5f}"))
        if d["val_mel_regressions"]:
            # Only flag regressions that were not subsequently recovered.
            # A regression at list-index i (stored ep = i+2) is recovered if
            # any later val_mel value falls at or below the pre-regression value.
            unrecovered = [
                (ep, dd) for ep, dd in d["val_mel_regressions"]
                if not any(v <= vm[ep - 2] for v in vm[ep:])
            ]
            # Separate SpecAugment-onset regressions from real ones
            sa_unrec = [(ep, dd) for ep, dd in unrecovered
                        if _is_spec_augment_transient(ep, sa_epoch, vm_vals=vm)]
            real_unrec = [(ep, dd) for ep, dd in unrecovered
                         if not _is_spec_augment_transient(ep, sa_epoch, vm_vals=vm)]
            if real_unrec:
                msg = "val_mel increased at epochs: " + ", ".join(
                    f"Ep{ep}+Δ{dd:+.5f}" for ep, dd in real_unrec)
                if sa_unrec:
                    msg += "  (+ " + ", ".join(
                        f"Ep{ep}" for ep, _ in sa_unrec) + " from SpecAugment onset)"
                flags.append(("WARN", msg))
            elif sa_unrec:
                flags.append(("PASS",
                    "val_mel increases at " + ", ".join(
                        f"Ep{ep}+Δ{dd:+.5f}" for ep, dd in sa_unrec)
                    + " — SpecAugment adaptation transient (not regression)"))
            else:
                rstr = ", ".join(f"Ep{ep}Δ{dd:+.5f}" for ep, dd in d["val_mel_regressions"])
                flags.append(("PASS", f"val_mel transient blip(s) at {rstr} — fully recovered"))

    # 1b. Gradual drift (new — catches slow upward slope even without individual bad epochs)
    if d.get("val_mel_gradual_drift"):
        flags.append(("WARN",
                      f"val_mel gradual upward drift: slope={d['val_mel_slope']:+.6f}/ep  "
                      f"R²={d['val_mel_r2']:.3f}  — no single bad epoch but sustained rise"))

    # 2. Val/train mel gap
    gap = d["val_train_gap"]
    if gap is not None:
        if gap > 1.0:
            flags.append(("FAIL", f"Val/train mel gap = {gap:.4f} — overfitting"))
        elif gap > 0.5:
            flags.append(("WARN", f"Val/train mel gap = {gap:.4f} at last epoch (mild overfitting)"))
        elif gap < -0.4:
            # Val is very much better than train — could indicate val set is
            # too small/easy, or dropout is very high.
            flags.append(("WARN", f"Val/train mel gap = {gap:.4f} (val << train) — "
                          "unusually large; check val set size/diversity or dropout level"))
        elif gap < 0.0:
            # Val < train is normal for non-autoregressive TTS: dropout is
            # active during training but not evaluation, so train loss is
            # artificially inflated relative to val.
            flags.append(("PASS", f"Val/train mel gap = {gap:.4f} "
                          "(val < train — expected for TTS with dropout)"))
        else:
            flags.append(("PASS", f"Val/train mel gap = {gap:.4f}"))

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

    # 4. Late grad spikes
    gn = _get(ea, "stats/grad_norm")
    if gn:
        step_range   = gn[-1][0] - gn[0][0]
        early_cutoff = gn[0][0] + step_range * 0.10
        late_spikes  = [v for s, v in gn if s > early_cutoff and v > 5.0]
        if len(late_spikes) > 10:
            flags.append(("FAIL", f"{len(late_spikes)} grad spikes >5.0 after init warmup"))
        elif len(late_spikes) > 3:
            flags.append(("WARN", f"{len(late_spikes)} late grad spikes >5.0"))
        else:
            flags.append(("PASS", f"Late grad spikes >5.0: {len(late_spikes)}"))

    # 5. Stop token balance
    ratio = d["stop_mel_ratio"]
    if ratio is not None:
        if ratio > 0.5:
            flags.append(("WARN", f"val_stop/val_mel ratio = {ratio:.3f} — stop still dominates"))
        else:
            flags.append(("PASS", f"val_stop/val_mel ratio = {ratio:.3f}"))

    # 6. LR ramp coupling
    vm_pairs = list(zip(range(1, len(vm) + 1), vm))
    if vm_pairs and len(vm_pairs) >= 3:
        dec = _get(ea, "stats/lr_decoder")
        if dec:
            max_lr     = max(v for _, v in dec)
            peak_step  = next((s for s, v in dec if abs(v - max_lr) < max_lr * 0.01), None)
            if peak_step:
                vm_series = _get(ea, "loss/val_mel_epoch")
                rpp       = [(s, v) for s, v in vm_series if s > peak_step]
                # Exclude post-peak data that falls within the SpecAugment
                # adaptation window — those increases are expected and transient.
                if sa_epoch is not None:
                    steps_per_ep = vm_series[1][0] - vm_series[0][0] if len(vm_series) >= 2 else 338
                    sa_step = (sa_epoch - 1) * steps_per_ep
                    rpp_filtered = [(s, v) for s, v in rpp if s < sa_step]
                    if len(rpp_filtered) >= 2:
                        rpp = rpp_filtered
                if len(rpp) >= 2:
                    trend = rpp[-1][1] - rpp[0][1]
                    if trend > 0.05:
                        flags.append(("WARN", f"val_mel UP post LR warmup peak (step {peak_step}): Δ={trend:+.5f}"))
                    else:
                        flags.append(("PASS", f"val_mel stable/down post LR warmup peak (step {peak_step})"))

    for lbl, msg in flags:
        icon = "✓" if lbl == "PASS" else ("⚠" if lbl == "WARN" else "✗")
        print(f"  [{lbl:4s}] {icon}  {msg}")


def analyze_loss_imbalance(ea, records=None):
    """
    Investigate gradient imbalance across layer categories.
    BUG FIX: previously called load_checkpoints() without compute_weight_stats,
    so param_stats was always empty.  Now ensures stats exist before proceeding,
    and uses the shared classify_param helper.
    """
    findings = []

    if records is None:
        try:
            records = load_checkpoints()
            records = compute_weight_stats(records)
        except Exception:
            records = []

    if not records or len(records) < 2:
        return ["Insufficient checkpoint data for gradient imbalance analysis."]

    # Ensure param_stats are present
    if not records[-1].get("param_stats"):
        try:
            records = compute_weight_stats(records)
        except Exception as e:
            return [f"Failed to compute parameter stats: {e}"]

    rec_b   = records[-1]
    rec_a   = records[-2]
    steps_b = rec_b.get("optimizer_steps")
    steps_a = rec_a.get("optimizer_steps")
    if not (isinstance(steps_b, int) and isinstance(steps_a, int) and steps_b > steps_a):
        return ["Optimizer step info missing or invalid in the last checkpoint pair."]

    ep_steps = steps_b - steps_a
    lr_series = _get(ea, "stats/lr_decoder")
    if not lr_series:
        return ["No lr_decoder series in TB logs — cannot estimate avg gradients."]

    in_range  = [v for s, v in lr_series if steps_a <= s <= steps_b]
    avg_lr_dec = statistics.mean(in_range) if in_range else max(v for _, v in lr_series)

    cfg = rec_b.get("config")
    enc_lr_mult = getattr(cfg, 'encoder_lr_multiplier', 1.3) if cfg else 1.3
    ffn_lr_mult = getattr(cfg, 'decoder_ffn_lr_multiplier', 1.0) if cfg else 1.0
    ffn_lr_tb   = _get(ea, "stats/lr_decoder_ffn")
    in_range_ffn = [v for s, v in ffn_lr_tb if steps_a <= s <= steps_b] if ffn_lr_tb else []
    avg_lr_ffn   = statistics.mean(in_range_ffn) if in_range_ffn else avg_lr_dec * ffn_lr_mult
    lr_map = {'decoder': avg_lr_dec, 'encoder': avg_lr_dec * enc_lr_mult,
              'decoder_ffn': avg_lr_ffn}

    ps = rec_b.get('param_stats', {})
    sd = rec_b.get('state_dict', {})
    # Two parallel tallies:
    #   cat_delta_sums  — total |Δw|_L2 per category (numel-weighted; large tensors dominate)
    #   cat_rms_sums    — total avg_grad_rms per category (per-element; useful for cap comparison)
    cat_delta_sums = {}
    cat_rms_sums   = {}
    total_delta    = 0.0
    per_tensor     = []   # (name, cat, avg_grad_rms, cap, numel, delta)
    unmatched      = []   # (name, avg_grad_rms, delta) for 'other' bucket

    for name, s in ps.items():
        delta = s.get('delta')
        if delta is None:
            continue
        cat, cap, lr_key = classify_param(name, cfg)
        lr_t = lr_map.get(lr_key, avg_lr_dec)
        if not lr_t or lr_t <= 0:
            continue
        tensor = sd.get(name)
        numel  = (tensor.numel() if tensor is not None
                  else max(1, int(s["norm"] ** 2 / max(s["mean"] ** 2, 1e-12))))
        avg_grad_rms = (delta / math.sqrt(numel)) / (ep_steps * lr_t)
        per_tensor.append((name, cat, avg_grad_rms, cap, numel, delta))
        cat_delta_sums[cat] = cat_delta_sums.get(cat, 0.0) + delta
        cat_rms_sums[cat]   = cat_rms_sums.get(cat, 0.0)  + abs(avg_grad_rms)
        total_delta         += delta
        if cat == 'other':
            unmatched.append((name, avg_grad_rms, delta))

    if not per_tensor:
        return ["No parameter deltas available between last two checkpoints."]

    # Share by total delta (numel-weighted) — the meaningful metric for "which group moves most"
    cat_share   = {c: 100.0 * v / total_delta for c, v in cat_delta_sums.items()} if total_delta > 0 else {}
    sorted_cats = sorted(cat_share.items(), key=lambda x: x[1], reverse=True)

    findings.append("Weight-delta shares by category (|Δw|_L2, last checkpoint pair):")
    for c, share in sorted_cats:
        findings.append(f"  - {c}: {share:.1f}%")

    dominant = [c for c, s in sorted_cats if s > 30.0 and c != 'other']
    if dominant:
        findings.append(f"Dominant weight-delta contributor(s): {', '.join(dominant)}")
    else:
        findings.append("No single named category >30% — activity spread across layers.")

    stop_share = cat_share.get('stop', 0.0)
    if stop_share > 15.0:
        findings.append(f"Stop-head contributes {stop_share:.1f}% of weight delta — "
                        "consider lowering stop pos_weight or smoothing targets.")

    # Flag tensors over their per-element cap (per-element RMS units, correct for comparison)
    over_cap_by_cat   = defaultdict(list)
    over2x_cap_by_cat = defaultdict(list)
    for name, cat, avg_grad_rms, cap, numel, delta in per_tensor:
        if cap == float('inf'):
            continue
        if avg_grad_rms >= 2 * cap:
            over2x_cap_by_cat[cat].append((name, avg_grad_rms, cap))
        elif avg_grad_rms >= cap:
            over_cap_by_cat[cat].append((name, avg_grad_rms, cap))

    if over2x_cap_by_cat:
        findings.append("SEVERE: tensors with avg_grad_rms ≥ 2× cap (likely driving spikes):")
        for cat, items in over2x_cap_by_cat.items():
            findings.append(f"  - {cat}: {len(items)} tensors  "
                            f"(worst: {items[0][0]}  rms={items[0][1]:.4f}  cap={items[0][2]:.1f}  "
                            f"={items[0][1]/items[0][2]:.1f}×cap)")
    elif over_cap_by_cat:
        findings.append("Tensors at or above their per-element cap:")
        for cat, items in over_cap_by_cat.items():
            findings.append(f"  - {cat}: {len(items)} tensors  "
                            f"(example: {items[0][0]}  rms={items[0][1]:.4f}  cap={items[0][2]:.1f})")

    # 'other' breakdown — show top contributors so they can be given explicit rules
    other_share = cat_share.get('other', 0.0)
    if unmatched and other_share > 5.0:
        unmatched.sort(key=lambda x: x[2], reverse=True)   # sort by delta, not RMS
        findings.append(f"'Other' bucket is {other_share:.1f}% of weight-delta activity "
                        f"({len(unmatched)} unclassified params). Top contributors:")
        for nm, g, d in unmatched[:8]:
            findings.append(f"  {nm:<60}  Δ={d:.4f}  rms={g:.4f}")
        if other_share > 15.0:
            findings.append("  → Add explicit classify_param() rules for the names above.")

    return findings


def tb_print_recommendations(ea, records=None):
    """
    Prioritised action list.  Now surfaces gradual-degradation recommendations
    in addition to spike-based ones, and correctly passes records through to
    analyze_loss_imbalance (fixing the empty-param_stats bug).
    """
    print("\n" + "=" * 90)
    print("TENSORBOARD — Analysis & Recommendations")
    print("=" * 90)

    d   = _collect_diagnostics(ea)
    cfg = _get_cfg_from_records(records)
    sa_epoch = _spec_augment_display_epoch(cfg)

    def cfg_get(attr, default):
        return getattr(cfg, attr, default) if cfg is not None else default

    cur_max_lr_mult     = cfg_get('max_lr_multiplier',         1.1)
    cur_enc_lr_mult     = cfg_get('encoder_lr_multiplier',     1.3)
    cur_stop_head_mult  = cfg_get('stop_head_lr_multiplier',   0.1)
    cur_stop_pos        = cfg_get('stop_token_pos_weight',    25.0)
    cur_stop_loss       = cfg_get('stop_token_loss_weight',   0.010)
    cur_stop_clip       = cfg_get('stop_head_spike_clip_norm', 0.5)
    cur_warmup          = cfg_get('warmup_steps',             1200)
    cur_weight_decay    = cfg_get('weight_decay',             0.01)
    cur_max_frames      = cfg_get('max_frames_per_batch',    15000)

    suggest_max_lr_mult  = max(0.1, cur_max_lr_mult  - 0.1)
    suggest_enc_lr_mult  = max(0.1, cur_enc_lr_mult  - 0.2)
    suggest_stop_pos     = max(1.0, int(cur_stop_pos  * 0.7))
    suggest_stop_loss    = max(0.001, cur_stop_loss   * 0.67)
    suggest_stop_head_mult = max(0.01, cur_stop_head_mult * 0.5)
    suggest_warmup       = int(cur_warmup + 400)
    suggest_weight_decay = cur_weight_decay * 2.0
    suggest_max_frames   = max(1000, int(cur_max_frames * 0.85))

    recs = []  # (priority, label, lines)  1=CRITICAL 2=WARN 3=INFO

    # ── Stop head isolation status ───────────────────────────────────────────
    # Check whether the stop head LR group is active and at a sensible ratio
    stop_lr_series = _get(ea, "stats/lr_stop_head")
    dec_lr_series  = _get(ea, "stats/lr_decoder")
    if stop_lr_series and dec_lr_series:
        stop_lr_curr = stop_lr_series[-1][1]
        dec_lr_curr  = dec_lr_series[-1][1]
        actual_mult  = stop_lr_curr / dec_lr_curr if dec_lr_curr > 0 else 0.0
        if abs(actual_mult - cur_stop_head_mult) < 0.01:
            recs.append((3, "INFO", [
                f"Stop head isolation active: LR multiply={actual_mult:.3f}× decoder  "
                f"(cfg stop_head_lr_multiplier={cur_stop_head_mult}).",
                f"  Gradient detach: decoder_outputs.detach() isolates mel decoder from BCE stop loss.",
                f"  Pre-clip cap: {cur_stop_clip} (stop_head_spike_clip_norm).",
            ]))
        elif actual_mult > cur_stop_head_mult * 2:
            recs.append((2, "WARN", [
                f"Stop head LR ratio ({actual_mult:.3f}×) >> configured stop_head_lr_multiplier "
                f"({cur_stop_head_mult}).  Checkpoint optimizer state may be stale.",
                f"    → Verify optimizer resumed correctly (expect 4 param groups).",
            ]))
    elif not stop_lr_series:
        recs.append((3, "INFO", [
            "stats/lr_stop_head not found in TB — stop head may be sharing the decoder LR group "
            "(pre-isolation checkpoint).  The dedicated param group becomes active next run.",
        ]))

    # ── Gradual degradation (NEW) ────────────────────────────────────────────
    vm = d["val_mel_series"]
    if d.get("val_mel_gradual_drift") and not d["val_mel_regressed"]:
        # Check if the drift is primarily caused by SpecAugment onset
        best_ep = vm.index(min(vm)) + 1 if len(vm) >= 3 else None
        drift_from_specaug = (
            sa_epoch is not None
            and best_ep is not None
            and best_ep < sa_epoch
        )
        if drift_from_specaug:
            # SpecAugment explains the drift — downgrade to INFO
            slope = d["val_mel_slope"]
            r2    = d["val_mel_r2"]
            recs.append((3, "INFO", [
                f"val_mel upward drift (slope={slope:+.6f}/ep  R²={r2:.3f}) coincides with "
                f"SpecAugment activation at Ep{sa_epoch:02d}.",
                f"  Best was Ep{best_ep:02d} (pre-augment). This is an expected adaptation transient.",
                "  Monitor: val_mel should plateau and begin recovering within 3–5 epochs.",
            ]))
        else:
            slope = d["val_mel_slope"]
            r2    = d["val_mel_r2"]
            body  = [
                f"val_mel shows sustained upward drift: slope={slope:+.6f}/ep  R²={r2:.3f}.",
                "  No single epoch looks alarming, but the trend is real.",
            ]
            if len(vm) >= 3:
                age = len(vm) - best_ep if best_ep else 0
                if age >= 2:
                    body.append(f"  Best checkpoint was Ep{best_ep:02d}, {age} epochs ago — "
                                 "model has not improved since.")
            gap = d["val_train_gap"]
            if gap is not None and gap > 0.4:
                body.append(f"  Val/train gap = {gap:.4f} — overfitting is a co-driver.")
                body.append(f"    → Increase weight_decay "
                            f"({cur_weight_decay:.4f} → {suggest_weight_decay:.4f}).")
            body.append(f"    → Reduce max_lr_multiplier "
                        f"({cur_max_lr_mult:.2f} → {suggest_max_lr_mult:.2f}) to soften weight updates.")
            body.append("    → Consider rolling back to the best checkpoint and resuming from there.")
            recs.append((1 if r2 > 0.6 else 2, "CRITICAL" if r2 > 0.6 else "WARN", body))

    # ── Val mel overall regression (existing) ───────────────────────────────
    if d["val_mel_regressed"]:
        body = [f"val_mel is HIGHER at last epoch ({vm[-1]:.5f}) vs first ({vm[0]:.5f})."]
        if d["val_train_gap"] is not None and d["val_train_gap"] > 0.8:
            body.append(f"  + Large val/train gap ({d['val_train_gap']:.4f}) — overfitting.")
            body.append(f"    → Increase weight_decay "
                        f"({cur_weight_decay:.4f} → {suggest_weight_decay:.4f}).")
        if d["clip_sat_pct"] > 40:
            body.append("  + Clipping saturation coincides — gradient pressure is a co-driver.")
            body.append(f"    → Reduce max_lr_multiplier "
                        f"({cur_max_lr_mult:.2f} → {suggest_max_lr_mult:.2f}).")
        if d["val_mel_regressions"]:
            ep_list = ", ".join(f"Ep{ep}" for ep, _ in d["val_mel_regressions"])
            body.append(f"  Regression epochs: {ep_list}.")
        recs.append((1, "CRITICAL", body))

    elif d["val_mel_regressions"]:
        # Filter to regressions not yet recovered (same logic as flag summary).
        unrecovered_recs = [
            (ep, dd) for ep, dd in d["val_mel_regressions"]
            if not any(v <= vm[ep - 2] for v in vm[ep:])
        ]
        # Exclude SpecAugment-onset transients from escalation logic
        real_unrec = [
            (ep, dd) for ep, dd in unrecovered_recs
            if not _is_spec_augment_transient(ep, sa_epoch, vm_vals=vm)
        ]
        sa_unrec = [
            (ep, dd) for ep, dd in unrecovered_recs
            if _is_spec_augment_transient(ep, sa_epoch, vm_vals=vm)
        ]
        if sa_unrec and not real_unrec:
            # All unrecovered regressions are from SpecAugment onset — INFO only
            ep_list = ", ".join(f"Ep{ep} Δ{dd:+.4f}" for ep, dd in sa_unrec)
            recs.append((3, "INFO", [
                f"val_mel increased at: {ep_list}.",
                "  These coincide with SpecAugment activation — expected adaptation transient.",
                "  Monitor: should plateau within 3–5 epochs as model adapts to masked inputs.",
            ]))
        elif real_unrec:
            ep_list = ", ".join(f"Ep{ep} Δ{dd:+.4f}" for ep, dd in real_unrec)
            if sa_unrec:
                ep_list += "  (+ " + ", ".join(
                    f"Ep{ep}" for ep, _ in sa_unrec) + " from SpecAugment)"
            n_consec = len(real_unrec)
            # Detect whether the regression strand has ended:
            # if any non-SA epoch after the last regression showed improvement.
            last_reg_ep = real_unrec[-1][0]  # 1-indexed
            strand_ended = False
            is_decelerating = False
            lr_ascending = d.get("lr_still_rising", False)
            for ep_1 in range(last_reg_ep + 1, len(vm) + 1):
                if _is_spec_augment_transient(ep_1, sa_epoch, vm_vals=vm):
                    continue
                if vm[ep_1 - 1] <= vm[ep_1 - 2]:
                    strand_ended = True
                    break
            if strand_ended:
                action = (
                    f"  Regression strand (Ep{real_unrec[0][0]}–{last_reg_ep}) has ended — "
                    f"subsequent epochs stabilised.\n"
                    f"  val_mel has not recovered to best ({min(vm):.5f} at "
                    f"Ep{vm.index(min(vm))+1:02d}).\n"
                    "  No immediate action needed; monitor post-SpecAugment recovery."
                ) if sa_epoch else (
                    f"  Regression strand (Ep{real_unrec[0][0]}–{last_reg_ep}) has ended — "
                    f"subsequent epochs stabilised.\n"
                    f"  val_mel has not recovered to best ({min(vm):.5f} at "
                    f"Ep{vm.index(min(vm))+1:02d}).\n"
                    "  Monitor for further recovery."
                )
            elif n_consec >= 2:
                last_delta = abs(real_unrec[-1][1])
                # Detect decelerating regression during LR ascent —
                # if each successive delta is smaller than the previous,
                # the regression is self-correcting (expected OneCycle behaviour).
                deltas = [abs(dd) for _, dd in real_unrec]
                is_decelerating = (
                    len(deltas) >= 2
                    and all(deltas[i] < deltas[i - 1] for i in range(1, len(deltas)))
                )
                if is_decelerating and lr_ascending:
                    decel_pct = 100.0 * (1.0 - deltas[-1] / deltas[0])
                    projected = deltas[-1] * (deltas[-1] / deltas[-2]) if deltas[-2] > 0 else 0
                    action = (
                        f"  Regression is decelerating ({decel_pct:.0f}% smaller than first).\n"
                        f"  LR still ascending — this is expected OneCycle cosine-ascent pressure.\n"
                        f"  Projected next Δ: ~{projected:+.4f} (should stabilise within 2-3 epochs).\n"
                        "  No intervention needed. Continue training."
                    )
                elif last_delta >= 0.010:
                    action = (
                        "  2nd consecutive rise ≥ 0.010 — ACT NOW (do not wait for next epoch):\n"
                        "    → If encoder FFN or decoder attention are new top movers: reduce max_lr_multiplier by 0.1\n"
                        "    → If decoder FFN still sole dominant mover: lower decoder_ffn_lr_multiplier by 0.1\n"
                        "    → Resume from best checkpoint."
                    )
                else:
                    action = (
                        "  2+ consecutive regressions — if next epoch also rises:\n"
                        "    → If decoder FFN is a persistent mover: lower decoder_ffn_lr_multiplier by 0.1\n"
                        "    → Otherwise: reduce max_lr_multiplier by 0.1\n"
                        "    → Consider resuming from best checkpoint."
                    )
            else:
                action = (
                    "  Monitor closely. If it continues next epoch:\n"
                    "    → lower decoder_ffn_lr_multiplier by 0.1 (if FFN layers are persistent movers)\n"
                    "    → or reduce max_lr_multiplier by 0.1"
                )
            is_lr_decel = is_decelerating and lr_ascending and not strand_ended
            severity = 3 if (strand_ended or is_lr_decel) else 2
            label = "INFO" if (strand_ended or is_lr_decel) else "WARN"
            recs.append((severity, label, [
                f"val_mel regressed at: {ep_list}.",
                action,
            ]))

    # ── Grad spikes (existing) ───────────────────────────────────────────────
    n_late = len(d["late_spikes"])
    n_gt10 = len(d["late_spikes_gt10"])
    n_gt20 = len(d["late_spikes_gt20"])
    co_pct = d["spike_stop_cooccur_pct"]
    n_ramp = len(d["spikes_during_ramp"])

    if n_gt20 > 0:
        body = [f"{n_gt20} spike(s) exceeded 20× — severe outlier batches.",
                f"  Max raw grad_norm: {d['spike_max']:.2f}."]
        if co_pct > 50:
            body.append(f"  {co_pct:.0f}% of spikes co-occur with elevated stop loss.")
            body.append("    → Add temporal smoothing to stop targets (soft labels near EOS).")
        recs.append((1, "CRITICAL", body))
    elif n_gt10 > 3:
        body = [f"{n_gt10} late spike(s) exceeded 10× (max {d['spike_max']:.2f})."]
        if co_pct > 40:
            body.append(f"  {co_pct:.0f}% co-occur with stop bursts.")
            body.append(f"    → Reduce stop_token_pos_weight ({cur_stop_pos:.0f} → {suggest_stop_pos:.0f}).")
            if stop_lr_series is None:
                body.append("    → stop_head_lr_multiplier group not yet active — ensure 4-group optimizer is loaded.")
            else:
                body.append(f"    → stop head LR group active at {cur_stop_head_mult}×; "
                             f"further reduce to {suggest_stop_head_mult:.3f} if spikes persist.")
        if n_ramp > len(d["spikes_post_peak"]) and d["lr_phase"] in ("warmup", "peak"):
            body.append(f"    → Increase warmup_steps ({cur_warmup} → {suggest_warmup}).")
        recs.append((2, "WARN", body))
    elif n_late > 10:
        recs.append((2, "WARN", [
            f"{n_late} late spikes >5× — above normal tolerance.",
            f"  max={d['spike_max']:.2f}  clusters={d['spike_clusters']}",
            "    → Watch saturation trend next epoch.",
        ]))
    elif n_late > 3:
        recs.append((3, "INFO", [
            f"{n_late} late spikes >5× — within normal tolerance.",
            f"  max={d['spike_max']:.2f}  clusters={d['spike_clusters']}",
        ]))

    # ── Clipping saturation (existing) ──────────────────────────────────────
    sat   = d["clip_sat_pct"]
    trend = d["clip_sat_trend"]
    if sat > 40 and trend != "improving":
        body = [
            f"Clipping saturation {sat:.1f}% — model taking truncated steps on over 40% of batches.",
            f"  Trend: {trend}.",
            f"    → Reduce max_lr_multiplier ({cur_max_lr_mult:.2f} → {suggest_max_lr_mult:.2f}).",
        ]
        recs.append((1, "CRITICAL", body))
    elif sat > 20 and trend == "worsening":
        recs.append((2, "WARN", [
            f"Clipping saturation {sat:.1f}% and worsening.",
            f"  Epochs: {[f'{p:.1f}%' for p in d['clip_sat_per_epoch']]}",
        ]))
    elif sat > 20:
        recs.append((3, "INFO", [
            f"Clipping saturation {sat:.1f}% (threshold 20%). Trend: {trend}.",
        ]))

    # ── Val/train gap ────────────────────────────────────────────────────────
    gap = d["val_train_gap"]
    if gap is not None and gap > 1.0 and not d["val_mel_regressed"]:
        recs.append((2, "WARN", [
            f"Val/train mel gap = {gap:.4f} — overfitting signal.",
            f"    → Increase weight_decay ({cur_weight_decay:.4f} → {suggest_weight_decay:.4f}).",
        ]))
    elif gap is not None and gap > 0.5 and not d["val_mel_regressed"]:
        recs.append((3, "INFO", [f"Val/train mel gap = {gap:.4f} (mild)."]))

    # ── Stop dominance ───────────────────────────────────────────────────────
    ratio = d["stop_mel_ratio"]
    if ratio is not None and ratio > 0.5:
        body = [f"val_stop/val_mel ratio = {ratio:.3f} — stop loss disproportionate."]
        if d["stop_late_bursts"]:
            body.append(f"  Late stop bursts detected: {d['stop_late_bursts'][:4]}")
        body.append(f"    → Reduce stop_token_pos_weight ({cur_stop_pos:.0f} → {suggest_stop_pos:.0f}).")
        if stop_lr_series is not None:
            body.append(f"    → stop head LR isolation active ({cur_stop_head_mult}×); "
                         f"consider reducing further to {suggest_stop_head_mult:.3f} if ratio persists.")
        recs.append((2, "WARN", body))

    # ── Attach imbalance diagnostics to CRITICAL items ───────────────────────
    critical = [r for r in recs if r[0] == 1]
    if critical:
        imbalance = analyze_loss_imbalance(ea, records=records)
        if imbalance:
            for idx, (pri, label, body) in enumerate(recs):
                if pri == 1:
                    body.append("")
                    body.append("  LOSS-IMBALANCE DIAGNOSTICS:")
                    for line in imbalance:
                        body.append(f"    {line}")

    warns = [r for r in recs if r[0] == 2]
    infos = [r for r in recs if r[0] == 3]

    if not critical and not warns:
        print("  ✓  No action items. Training is progressing normally.")
    elif critical:
        print(f"  {len(critical)} CRITICAL  {len(warns)} WARN  {len(infos)} INFO\n")
    else:
        print(f"  {len(warns)} WARN  {len(infos)} INFO\n")

    priority_labels = {1: "CRITICAL", 2: "WARN    ", 3: "INFO    "}
    icons           = {1: "✗", 2: "⚠", 3: "ℹ"}

    for idx, (pri, label, lines) in enumerate(recs, 1):
        print(f"  [{idx}] {icons[pri]} {priority_labels[pri]}  {lines[0]}")
        for line in lines[1:]:
            print(f"            {line}")
        print()


def tb_print_lr_phase_detail(ea, records=None):
    lr  = _get(ea, "stats/lr_decoder")
    enc = _get(ea, "stats/lr_encoder")
    if not lr:
        return

    true_peak_dec = true_peak_enc = abs_peak_step = None
    cfg_note = "(config unavailable — falling back to observed max)"

    if records:
        last = max(records, key=lambda r: r["epoch_num"])
        cfg  = last.get("config")
        if cfg is not None:
            learning_rate = getattr(cfg, 'learning_rate',         1e-4)
            max_lr_mult   = getattr(cfg, 'max_lr_multiplier',     1.1)
            enc_lr_mult   = getattr(cfg, 'encoder_lr_multiplier', 1.3)
            pct_start     = getattr(cfg, 'pct_start',             0.2)
            use_warmup    = getattr(cfg, 'use_warmup',            True)
            warmup_steps  = getattr(cfg, 'warmup_steps',         1200) if use_warmup else 0
            num_epochs    = getattr(cfg, 'num_epochs',             100)
            opt_steps     = last.get("optimizer_steps")
            ep_num        = last.get("epoch_num")
            if isinstance(opt_steps, int) and isinstance(ep_num, int) and ep_num > 0:
                steps_per_epoch = opt_steps // ep_num
                total_steps     = num_epochs * steps_per_epoch
                if warmup_steps >= total_steps:
                    warmup_steps = max(0, total_steps - 1)
                onecycle_steps = max(1, total_steps - warmup_steps)
                peak_onecycle  = int(pct_start * onecycle_steps)
                abs_peak_step  = warmup_steps + peak_onecycle
                true_peak_dec  = learning_rate * max_lr_mult
                true_peak_enc  = learning_rate * max_lr_mult * enc_lr_mult
                cfg_note       = (f"lr={learning_rate:.2e}  ×  max_lr_mult={max_lr_mult}  "
                                  f"steps/epoch={steps_per_epoch}  warmup={warmup_steps}  "
                                  f"onecycle={onecycle_steps}  pct_start={pct_start}")

    obs_max_dec  = max(v for _, v in lr)
    curr_lr      = lr[-1][1]
    curr_step    = lr[-1][0]
    ref_peak_dec = true_peak_dec if true_peak_dec is not None else obs_max_dec
    ref_peak_enc = (true_peak_enc if true_peak_enc is not None
                    else (max(v for _, v in enc) if enc else ref_peak_dec))

    tail         = [v for _, v in lr[-10:]]
    still_rising = len(tail) >= 2 and tail[-1] > tail[0]
    obs_peak_step = (next((s for s, v in lr if v >= obs_max_dec * 0.99), None)
                     if not still_rising else None)

    print("\n" + "=" * 90)
    print("TENSORBOARD — LR Phase Detail (from 90% of peak onward)")
    print("=" * 90)
    print(f"  schedule      : {cfg_note}")
    if true_peak_dec is not None:
        steps_to_peak = abs_peak_step - curr_step
        direction     = (f"{steps_to_peak:+d} steps to peak"
                         if steps_to_peak > 0 else f"{-steps_to_peak} steps past peak")
        print(f"  true peak LR  : {true_peak_dec:.8f} (dec)   {true_peak_enc:.8f} (enc)   "
              f"at step {abs_peak_step}  [{direction}]")
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

    # Decoder FFN LR (dedicated param group with decoder_ffn_lr_multiplier)
    ffn_series = _get(ea, "stats/lr_decoder_ffn")
    if ffn_series:
        ffn_lr_curr = ffn_series[-1][1]
        ffn_pct     = 100.0 * ffn_lr_curr / ref_peak_dec
        ffn_mult    = ffn_lr_curr / curr_lr if curr_lr > 0 else 0.0
        print(f"  decoder FFN LR: {ffn_lr_curr:.8f}  at step {ffn_series[-1][0]}  "
              f"({ffn_pct:.4f}% of dec peak  {ffn_mult:.3f}× decoder)")

    # Stop head LR (dedicated group, should track decoder × stop_head_lr_multiplier)
    stop_series = _get(ea, "stats/lr_stop_head")
    if stop_series:
        stop_lr_curr = stop_series[-1][1]
        stop_lr_max  = max(v for _, v in stop_series)
        stop_pct     = 100.0 * stop_lr_curr / ref_peak_dec
        stop_mult    = stop_lr_curr / curr_lr if curr_lr > 0 else 0.0
        print(f"  stop head LR  : {stop_lr_curr:.8f}  at step {stop_series[-1][0]}  "
              f"({stop_pct:.4f}% of dec peak  {stop_mult:.3f}× decoder)")
    else:
        print("  stop head LR  : no stats/lr_stop_head data (pre-isolation checkpoint or TB not refreshed)")

    thresh_90_step = next((s for s, v in lr if v >= obs_max_dec * 0.90), lr[0][0])
    print(f"\n  Step-by-step from step {thresh_90_step} onward:")
    print(f"  {'step':>6}  {'lr_decoder':>12}  {'% true peak':>12}  "
          f"{'lr_encoder':>12}  {'% true peak':>12}")
    print("  " + "-" * 66)

    enc_lookup   = {s: v for s, v in enc} if enc else {}
    flag_bucket  = ((abs_peak_step // 100) * 100 if abs_peak_step is not None
                    else ((obs_peak_step // 100) * 100 if obs_peak_step else None))
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
        enc_str = (f"{enc_v:.8f}  {100*enc_v/ref_peak_enc:>11.3f}%"
                   if enc_v else "              ?             ?")
        flag    = "  ← peak" if (flag_bucket is not None and bucket == flag_bucket) else ""
        print(f"  {s:>6}  {v:.8f}  {dec_pct:>11.3f}%{flag}  {enc_str}")

    if still_rising and abs_peak_step is not None and abs_peak_step > curr_step:
        proj_bucket = (abs_peak_step // 100) * 100
        if proj_bucket not in seen_buckets:
            enc_proj = (f"{true_peak_enc:.8f}  {'100.000%':>11}"
                        if true_peak_enc else "?")
            print(f"  {abs_peak_step:>6}  {true_peak_dec:.8f}  {'100.000%':>11}  "
                  f"← peak (projected)  {enc_proj}")


def tb_print_late_spike_context(ea):
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
    stop_p75  = _pct(stop, 75)   # safe — _pct handles empty series

    print(f"  {'step':>6}  {'raw grad':>9}  {'lr %peak':>9}  "
          f"{'stop nearby':>12}  {'stop elevated?':>15}  attribution")
    print("  " + "-" * 80)

    attrs = []
    for s, v in sorted(late_spikes):
        lr_here   = (min(lr_lookup.items(), key=lambda x: abs(x[0] - s))[1]
                     if lr_lookup else 0)
        lr_pct    = 100.0 * lr_here / lr_max if lr_max else 0
        stop_here = (min(stop, key=lambda x: abs(x[0] - s))[1]
                     if stop else 0)
        stop_flag = "YES ▲" if stop_here > stop_p75 else "no"

        if stop_here > stop_p75 and lr_pct < 97:
            attr = "stop burst"
        elif lr_pct >= 97 and stop_here <= stop_p75:
            attr = "LR at peak"
        elif lr_pct >= 97 and stop_here > stop_p75:
            attr = "LR peak + stop"
        else:
            attr = "outlier batch"
        attrs.append(attr)

        print(f"  {s:>6}  {v:>9.3f}  {lr_pct:>8.1f}%  "
              f"{stop_here:>12.5f}  {stop_flag:>15}  {attr}")

    counts = {k: attrs.count(k) for k in set(attrs)}
    print(f"\n  Attribution summary: "
          + "  ".join(f"{k}={n}" for k, n in sorted(counts.items())))


def tb_print_spike_cause_analysis(ea, records=None):
    """
    Gradient pre-clip pressure analysis.

    avg_grad_rms is computed as:
        (delta_L2 / sqrt(numel)) / (ep_steps * avg_lr)

    This gives a per-element RMS weight change per step — the same units as a
    per-element gradient magnitude, making it directly comparable to the
    per-element pre-clip caps used by the trainer.  The previous formulation
    omitted the sqrt(numel) denominator, producing values that were
    tensor-size-dependent and orders of magnitude too large.
    """
    if not records or len(records) < 2:
        return

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

    lr_series  = _get(ea, "stats/lr_decoder")
    enc_series = _get(ea, "stats/lr_encoder")
    if not lr_series:
        return

    in_range_dec = [v for s, v in lr_series if steps_a <= s <= steps_b]
    avg_lr_dec   = statistics.mean(in_range_dec) if in_range_dec else max(v for _, v in lr_series)

    if enc_series:
        in_range_enc = [v for s, v in enc_series if steps_a <= s <= steps_b]
        avg_lr_enc   = statistics.mean(in_range_enc) if in_range_enc else None
    else:
        avg_lr_enc = None

    ffn_series = _get(ea, "stats/lr_decoder_ffn")
    if ffn_series:
        in_range_ffn = [v for s, v in ffn_series if steps_a <= s <= steps_b]
        avg_lr_ffn   = statistics.mean(in_range_ffn) if in_range_ffn else None
    else:
        avg_lr_ffn = None

    cfg = rec_b.get("config")
    enc_lr_mult = getattr(cfg, 'encoder_lr_multiplier', 1.3) if cfg else 1.3
    ffn_lr_mult = getattr(cfg, 'decoder_ffn_lr_multiplier', 1.0) if cfg else 1.0
    if avg_lr_enc is None:
        avg_lr_enc = avg_lr_dec * enc_lr_mult
    if avg_lr_ffn is None:
        avg_lr_ffn = avg_lr_dec * ffn_lr_mult
    lr_map = {'decoder': avg_lr_dec, 'encoder': avg_lr_enc, 'decoder_ffn': avg_lr_ffn}

    groups   = defaultdict(list)   # cat -> [(name, avg_grad_rms, cap, numel)]
    ps       = rec_b.get("param_stats", {})
    sd       = rec_b.get("state_dict", {})
    unmatched = []  # (name, avg_grad_rms) for 'other' bucket

    for name, s in ps.items():
        delta = s.get("delta")
        if delta is None:
            continue
        cat, cap, lr_key = classify_param(name, cfg)
        lr_t = lr_map.get(lr_key, avg_lr_dec)
        if not lr_t or lr_t <= 0:
            continue
        # numel from the actual tensor when available, fall back to norm²/mean² estimate
        tensor = sd.get(name)
        numel  = tensor.numel() if tensor is not None else max(1, int(s["norm"] ** 2 / max(s["mean"] ** 2, 1e-12)))
        # per-element RMS change per step
        avg_grad_rms = (delta / math.sqrt(numel)) / (ep_steps * lr_t)
        groups[cat].append((name, avg_grad_rms, cap, numel))
        if cat == 'other':
            unmatched.append((name, avg_grad_rms, delta))

    if not groups:
        return

    gn = _get(ea, "stats/grad_norm")
    if gn:
        step_range   = gn[-1][0] - gn[0][0]
        early_cutoff = gn[0][0] + step_range * 0.10
        obs_max      = max((v for s, v in gn if s > early_cutoff), default=0.0)
    else:
        obs_max = 0.0

    print("\n" + "=" * 90)
    print("TENSORBOARD — Gradient Pre-Clip Pressure Analysis")
    print("=" * 90)
    print(f"  Checkpoint pair : {rec_a['file']}  →  {rec_b['file']}")
    print(f"  Epoch steps     : {ep_steps}")
    ffn_lr_str = f"   ffn: {avg_lr_ffn:.8f}" if avg_lr_ffn is not None else ""
    print(f"  Avg LR decoder  : {avg_lr_dec:.8f}   encoder: {avg_lr_enc:.8f}{ffn_lr_str}")
    print(f"  avg_grad_rms    = (Δ_L2 / √numel) / (steps × lr)   [per-element units, comparable to cap]")

    CAT_LABELS = {
        'dec_ffn':   'Decoder  FFN',
        'enc_ffn':   'Encoder  FFN',
        'dec_attn':  'Decoder Attn',
        'enc_attn':  'Encoder Attn',
        'mel_proj':  'Mel Proj    ',
        'stop':      'Stop Head   ',
        'variance':  'Variance Adp',
        'embedding': 'Embeddings  ',
        'layer_norm':'LayerNorm   ',
        'bias':      'Biases      ',
        'conv':      'Conv Layers ',
        'pos_enc':   'Pos Encoding',
        'other':     'Other(unkwn)',
    }
    CAT_ORDER = [
        'dec_ffn', 'enc_ffn', 'dec_attn', 'enc_attn',
        'mel_proj', 'stop', 'variance',
        'embedding', 'layer_norm', 'bias', 'conv', 'pos_enc', 'other',
    ]

    print()
    print(f"  {'Category':<14} {'N':>4}  {'min_rms':>9}  {'avg_rms':>9}  {'max_rms':>9}  "
          f"{'cap':>6}  {'×cap(avg)':>9}  {'≥cap':>6}  {'>>cap(2×)':>9}  flags")
    print("  " + "-" * 98)

    theoretical_max_sq = 0.0
    flags_list         = []
    total_rms_sq       = sum(g ** 2 for entries in groups.values() for _, g, _, _ in entries)

    for cat in CAT_ORDER:
        if cat not in groups:
            continue
        entries = groups[cat]
        grads   = [g for _, g, _, _ in entries]
        cap     = entries[0][2]
        n       = len(entries)
        min_g   = min(grads)
        avg_g   = statistics.mean(grads)
        max_g   = max(grads)

        # How many are at or above cap, and how many are >2× cap
        at_cap     = sum(1 for g in grads if cap != float('inf') and g >= cap)
        over2x_cap = sum(1 for g in grads if cap != float('inf') and g >= 2 * cap)
        ratio_str  = f"{avg_g / cap:.2f}×" if cap not in (0, float('inf')) else "   n/a"

        if cap != float('inf'):
            cosp               = math.sqrt(n * cap ** 2)
            theoretical_max_sq += n * cap ** 2
        else:
            cosp = math.sqrt(sum(g ** 2 for g in grads))

        # Severity flag
        flag = ""
        if cap == float('inf') and max_g > 1.0:
            flag = "UNCAPPED+ACTIVE"
            flags_list.append(
                f"  {CAT_LABELS.get(cat, cat)}: {n} uncapped tensors, "
                f"max_rms={max_g:.4f}  avg_rms={avg_g:.4f}")
        elif over2x_cap > 0:
            flag = f">>CAP ×{max_g/cap:.1f} ({over2x_cap}/{n})"
            flags_list.append(
                f"  {CAT_LABELS.get(cat, cat)}: {over2x_cap}/{n} tensors "
                f"avg_grad_rms ≥ 2× cap  "
                f"(max={max_g:.4f}  avg={avg_g:.4f}  cap={cap:.1f})"
                f"  ← likely exceeding pre-clip, driving spikes")
        elif at_cap > 0:
            flag = f"AT/OVER CAP ({at_cap}/{n})"
            flags_list.append(
                f"  {CAT_LABELS.get(cat, cat)}: {at_cap}/{n} tensors avg_grad_rms ≥ cap  "
                f"(max={max_g:.4f}  avg={avg_g:.4f}  cap={cap:.1f})")
        elif cap != float('inf') and avg_g >= cap * 0.5:
            flag = "NEAR CAP"
            flags_list.append(
                f"  {CAT_LABELS.get(cat, cat)}: avg_rms={avg_g:.4f} is "
                f"≥50% of cap={cap:.1f}")

        cap_str = f"{cap:.1f}" if cap != float('inf') else "    ∞"
        print(f"  {CAT_LABELS.get(cat, cat):<14} {n:>4}  {min_g:>9.4f}  {avg_g:>9.4f}  "
              f"{max_g:>9.4f}  {cap_str:>6}  {ratio_str:>9}  "
              f"{at_cap:>3}/{n:<3}  {over2x_cap:>5}/{n:<3}  {flag}")

    theoretical_max = math.sqrt(theoretical_max_sq)
    print()
    print(f"  Theoretical worst-case co-spike  √(Σ N·cap²) : {theoretical_max:.1f}")
    if obs_max > 0:
        print(f"  Observed max late spike in TB               : {obs_max:.2f}")
        headroom = theoretical_max / obs_max
        headroom_note = ("  ← pre-clips are the binding constraint" if headroom < 2
                         else "  (headroom OK)")
        print(f"  Headroom (theory / observed)                : {headroom:.1f}×{headroom_note}")

    if flags_list:
        print(f"\n  Flagged groups:")
        for fl in flags_list:
            print(fl)
    else:
        print(f"\n  All groups well below their pre-clip caps — gradient pressure nominal.")

    # Always show what's in the 'other' bucket so nothing is silently swallowed
    if unmatched:
        unmatched.sort(key=lambda x: x[2], reverse=True)   # sort by delta
        total_other_delta = sum(d for _, _, d in unmatched)
        total_all_delta   = sum(
            ps.get(name, {}).get('delta', 0.0) or 0.0
            for entries in groups.values() for name, _, _, _ in entries
        )
        other_pct = 100.0 * total_other_delta / total_all_delta if total_all_delta > 0 else 0.0
        print(f"\n  'Other' bucket breakdown  ({len(unmatched)} params, "
              f"{other_pct:.1f}% of total |Δw|):")
        print(f"  {'param name':<60}  {'|Δw|':>9}  {'avg_rms':>9}")
        print("  " + "-" * 82)
        for name, g, d in unmatched[:20]:
            print(f"  {name:<60}  {d:>9.4f}  {g:>9.4f}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")
        if other_pct > 15:
            print(f"\n  ⚠  'Other' is {other_pct:.1f}% of total weight-delta activity.")
            print(f"     Add explicit classify_param() rules for the names above to get accurate caps.")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level TB orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def tb_analyze(log_dir: Path = TB_LOG_DIR, records=None):
    ea = _load_tb(log_dir)
    if ea is None:
        return
    cfg = _get_cfg_from_records(records)
    print(f"\n[TensorBoard] Log dir: {log_dir}")
    tags = sorted(ea.Tags().get("scalars", []))
    print(f"[TensorBoard] {len(tags)} scalar tags available: {tags}")

    tb_print_step_loss_summary(ea)
    tb_print_val_mel_series(ea, cfg=cfg)
    tb_print_per_loss_trend(ea, cfg=cfg)
    tb_print_epoch_table(ea)
    tb_print_mel_stop_window_correlation(ea)
    tb_print_stop_token_analysis(ea, cfg=cfg)
    tb_print_gradient_analysis(ea)
    tb_print_late_spike_context(ea)
    tb_print_spike_cause_analysis(ea, records=records)
    tb_print_lr_trajectory(ea)
    tb_print_lr_phase_detail(ea, records=records)
    tb_print_regression_flags(ea, cfg=cfg)
    tb_print_recommendations(ea, records=records)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global CHECKPOINT_DIR, TB_LOG_DIR

    parser = argparse.ArgumentParser(
        description="Checkpoint + TensorBoard regression analysis."
    )
    parser.add_argument(
        "--model",
        default=str(CHECKPOINT_DIR),
        metavar="DIR",
        help=f"Path to the model directory (default: {CHECKPOINT_DIR})",
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
        # If TensorBoard logs exist, run TB-only analysis so users can still
        # inspect training behaviour even before the first checkpoint is saved.
        if TB_LOG_DIR.is_dir():
            print(f"No checkpoints found — falling back to TensorBoard logs at {TB_LOG_DIR} for analysis.")
            tb_analyze(TB_LOG_DIR, records=None)
            return
        else:
            print(f"No tensorboard logs found at: {TB_LOG_DIR}")
            return

    print(f"Found {len(records)} checkpoint(s). Computing weight statistics...")
    records = compute_weight_stats(records)
    records, persistent_counts = compute_rank_stability(records)

    print_summary_table(records)
    print_key_layer_table(records)
    print_key_layer_delta_table(records)
    print_biggest_deltas(records)
    print_persistent_movers(persistent_counts, n_epochs=sum(
        1 for r in records if r.get("rank_stability_jaccard") is not None))
    flag_regression_epoch(records)
    print_gradual_degradation_report(records)  # NEW

    bad = [r for r in records if r["any_nan"] or r["any_inf"]]
    if bad:
        print("\n!!! CHECKPOINTS WITH NaN or Inf WEIGHTS !!!")
        for r in bad:
            print(f"  {r['file']}  nan={r['any_nan']}  inf={r['any_inf']}")
    else:
        print("\nNo NaN or Inf weights found in any checkpoint.")

    tb_analyze(TB_LOG_DIR, records=records)


if __name__ == "__main__":
    main()
