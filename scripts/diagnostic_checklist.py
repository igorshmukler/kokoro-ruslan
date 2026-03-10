"""One-command PASS/FAIL diagnostic checklist for post-epoch-4 regression.

Checks implemented:
1) LR-to-loss coupling (rising decoder LR + rising mel/total)
2) Stop-vs-mel imbalance (effective stop contribution ratio R)
3) Clipping saturation (raw grad spikes + clipped-at-cap prevalence)
4) Validation divergence (stop improves while mel/spectral worsen)
5) Spike localization (grad spikes co-occur with loss/LR jumps)
6) Decision rule (>=3 checks fail => likely LR-ramp + stop-pressure regression)

Usage:
  python scripts/diagnostic_checklist.py
  python scripts/diagnostic_checklist.py --log-dir my_model/logs --baseline-epoch 4
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _series(ea: EventAccumulator, tag: str) -> List[Tuple[int, float]]:
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return []
    return [(e.step, float(e.value)) for e in ea.Scalars(tag)]


def _window(vals: List[Tuple[int, float]], start: int, end: int) -> List[Tuple[int, float]]:
    return [(s, v) for s, v in vals if start <= s <= end]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _nearest_at_or_before(series: List[Tuple[int, float]], step: int) -> Optional[float]:
    candidate = None
    for s, v in series:
        if s <= step:
            candidate = v
        else:
            break
    return candidate


def _next_after(series: List[Tuple[int, float]], step: int) -> Optional[float]:
    for s, v in series:
        if s > step:
            return v
    return None


def check_lr_loss_coupling(
    lr_decoder: List[Tuple[int, float]],
    mel: List[Tuple[int, float]],
    total: List[Tuple[int, float]],
    post_start: int,
) -> CheckResult:
    lr_w = [(s, v) for s, v in lr_decoder if s >= post_start]
    mel_w = [(s, v) for s, v in mel if s >= post_start]
    tot_w = [(s, v) for s, v in total if s >= post_start]

    if len(lr_w) < 8 or len(mel_w) < 8 or len(tot_w) < 8:
        return CheckResult("LR-to-loss coupling", False, "insufficient points after baseline")

    def qtrend(vals: List[Tuple[int, float]]) -> float:
        arr = [v for _, v in vals]
        q = max(1, len(arr) // 4)
        return (_mean(arr[-q:]) - _mean(arr[:q]))

    lr_delta = qtrend(lr_w)
    mel_delta = qtrend(mel_w)
    total_delta = qtrend(tot_w)

    # PASS when we do NOT see joint increase in LR and both losses
    coupled_rise = lr_delta > 0 and mel_delta > 0 and total_delta > 0
    passed = not coupled_rise
    return CheckResult(
        "LR-to-loss coupling",
        passed,
        f"Δlr={lr_delta:.3e}, Δmel={mel_delta:+.4f}, Δtotal={total_delta:+.4f}",
    )


def check_stop_mel_imbalance(
    stop: List[Tuple[int, float]],
    mel: List[Tuple[int, float]],
    stop_weight: float,
    post_start: int,
) -> CheckResult:
    stop_w = [(s, v) for s, v in stop if s >= post_start]
    mel_w = [(s, v) for s, v in mel if s >= post_start]
    if len(stop_w) < 8 or len(mel_w) < 8:
        return CheckResult("Stop-vs-mel imbalance", False, "insufficient points after baseline")

    # align by nearest mel value at/ before stop step
    ratios: List[float] = []
    for s, stop_v in stop_w:
        mel_v = _nearest_at_or_before(mel_w, s)
        if mel_v is None or mel_v <= 1e-9:
            continue
        ratios.append((stop_v * stop_weight) / mel_v)

    if len(ratios) < 8:
        return CheckResult("Stop-vs-mel imbalance", False, "insufficient aligned ratio points")

    q = max(1, len(ratios) // 4)
    r1 = _mean(ratios[:q])
    r4 = _mean(ratios[-q:])
    rise = r4 > r1 * 1.15
    # PASS when ratio is stable or decreasing
    passed = not rise
    return CheckResult("Stop-vs-mel imbalance", passed, f"R_q1={r1:.4f}, R_q4={r4:.4f}")


def check_clipping_saturation(
    grad: List[Tuple[int, float]],
    grad_clip: List[Tuple[int, float]],
    clip_cap: float,
    post_start: int,
) -> CheckResult:
    g = [(s, v) for s, v in grad if s >= post_start]
    c = [(s, v) for s, v in grad_clip if s >= post_start]
    if len(g) < 20 or len(c) < 20:
        return CheckResult("Clipping saturation", False, "insufficient grad points after baseline")

    spike_count = sum(1 for _, v in g if v > 5.0)
    sat_count = sum(1 for _, v in c if abs(v - clip_cap) <= max(1e-6, 0.01 * clip_cap))
    sat_ratio = sat_count / len(c)

    # FAIL if many spikes and clipping cap is frequently hit
    failing = spike_count >= 3 and sat_ratio >= 0.15
    passed = not failing
    return CheckResult(
        "Clipping saturation",
        passed,
        f"spikes>5={spike_count}, clip_cap_hits={sat_count}/{len(c)} ({sat_ratio:.1%})",
    )


def check_validation_divergence(
    val_mel_epoch: List[Tuple[int, float]],
    val_stop_epoch: List[Tuple[int, float]],
    val_sc_epoch: List[Tuple[int, float]],
) -> CheckResult:
    if len(val_mel_epoch) < 2 or len(val_stop_epoch) < 2 or len(val_sc_epoch) < 2:
        return CheckResult("Validation divergence", False, "insufficient epoch-level validation points")

    mel_prev, mel_last = val_mel_epoch[-2][1], val_mel_epoch[-1][1]
    stop_prev, stop_last = val_stop_epoch[-2][1], val_stop_epoch[-1][1]
    sc_prev, sc_last = val_sc_epoch[-2][1], val_sc_epoch[-1][1]

    # divergence pattern: stop improves (down), mel worsens (up), spectral convergence worsens (up)
    divergence = (stop_last < stop_prev) and (mel_last > mel_prev) and (sc_last > sc_prev)
    passed = not divergence
    return CheckResult(
        "Validation divergence",
        passed,
        f"Δval_stop={stop_last-stop_prev:+.4f}, Δval_mel={mel_last-mel_prev:+.4f}, Δval_SC={sc_last-sc_prev:+.4f}",
    )


def check_spike_localization(
    grad: List[Tuple[int, float]],
    stop: List[Tuple[int, float]],
    total: List[Tuple[int, float]],
    lr_decoder: List[Tuple[int, float]],
    post_start: int,
) -> CheckResult:
    grad_w = [(s, v) for s, v in grad if s >= post_start]
    spikes = [s for s, v in grad_w if v > 5.0]
    if not spikes:
        return CheckResult("Spike localization", True, "no grad spikes > 5.0")

    linked = 0
    checked = 0
    for s in spikes:
        checked += 1
        s0, s1 = s - 20, s + 20

        stop_vals = [v for ss, v in stop if s0 <= ss <= s1]
        total_vals = [v for ss, v in total if s0 <= ss <= s1]
        lr_prev = _nearest_at_or_before(lr_decoder, s0)
        lr_next = _next_after(lr_decoder, s1)

        jump_stop = False
        jump_total = False
        rise_lr = False

        if len(stop_vals) >= 2:
            jump_stop = (max(stop_vals) - min(stop_vals)) > 0.10
        if len(total_vals) >= 2:
            jump_total = (max(total_vals) - min(total_vals)) > 0.10
        if lr_prev is not None and lr_next is not None:
            rise_lr = (lr_next - lr_prev) > 0

        if jump_stop and jump_total and rise_lr:
            linked += 1

    frac = linked / checked if checked else 0.0
    # FAIL if majority of spikes are coupled to stop/total jumps under LR rise
    passed = frac < 0.5
    return CheckResult("Spike localization", passed, f"linked_spikes={linked}/{checked} ({frac:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="PASS/FAIL diagnostic checklist for post-epoch-4 regression")
    parser.add_argument("--log-dir", default="my_model/logs", help="TensorBoard log directory")
    parser.add_argument("--baseline-epoch", type=int, default=4, help="Epoch after which to evaluate regression behavior")
    parser.add_argument("--steps-per-epoch", type=int, default=641, help="Steps per epoch for windowing")
    parser.add_argument("--stop-weight", type=float, default=0.075, help="stop_token_loss_weight")
    parser.add_argument("--clip-cap", type=float, default=1.5, help="expected global grad clip cap")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        raise SystemExit(f"Log directory not found: {log_dir}")

    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()

    # Required step-level tags
    lr_decoder = _series(ea, "stats/lr_decoder")
    grad = _series(ea, "stats/grad_norm")
    grad_clip = _series(ea, "stats/grad_norm_clipped")
    mel = _series(ea, "loss/mel")
    total = _series(ea, "loss/total")
    stop = _series(ea, "loss/stop")

    # Required epoch-level tags
    val_mel_epoch = _series(ea, "loss/val_mel_epoch")
    val_stop_epoch = _series(ea, "loss/val_stop_epoch")
    val_sc_epoch = _series(ea, "metrics/val_spectral_convergence")

    post_start = args.baseline_epoch * args.steps_per_epoch + 1

    checks = [
        check_lr_loss_coupling(lr_decoder, mel, total, post_start),
        check_stop_mel_imbalance(stop, mel, args.stop_weight, post_start),
        check_clipping_saturation(grad, grad_clip, args.clip_cap, post_start),
        check_validation_divergence(val_mel_epoch, val_stop_epoch, val_sc_epoch),
        check_spike_localization(grad, stop, total, lr_decoder, post_start),
    ]

    fail_count = sum(1 for c in checks if not c.passed)
    decision_passed = fail_count < 3
    checks.append(
        CheckResult(
            "Decision rule",
            decision_passed,
            f"failed_checks={fail_count}/5 (threshold: >=3 => likely LR-ramp + stop-pressure)",
        )
    )

    print("\n=== Regression Diagnostic Checklist (PASS/FAIL) ===")
    print(f"log_dir={log_dir}  baseline_epoch={args.baseline_epoch}  post_start_step={post_start}")
    print("-" * 78)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"[{status}] {c.name}: {c.detail}")

    print("-" * 78)
    if decision_passed:
        print("Overall: PASS (regression pattern not strongly confirmed by this checklist)")
    else:
        print("Overall: FAIL (regression pattern is consistent with LR-ramp + stop-pressure)")


if __name__ == "__main__":
    main()
