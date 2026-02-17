# Training Convergence Checklist (Ruslan, MPS)

Use this checklist during training to decide **continue / tune / stop** early.

---

## 0) Run Baseline (fill once)

- [ ] Device confirmed: `mps`
- [ ] Precision mode confirmed in logs (actual mode, not requested mode)
- [ ] Scheduler confirmed: `OneCycleLR`
- [ ] Dataset + metadata version noted
- [ ] Checkpoint resume state noted (epoch/step)

## 1) Per-Epoch Health Checks

Complete at the end of every epoch:

- [ ] **No instability:** no NaN/Inf losses, no exploding gradients, no repeated backend errors
- [ ] **Memory health:** no OOM, memory state mostly `low`/`moderate`
- [ ] **Loss trend:** epoch-average `total_loss` is flat-to-down over 2–3 epochs (allow small noise)
- [ ] **Mel trend:** `mel_loss` is generally decreasing (main quality driver)
- [ ] **Auxiliary trends:** duration/pitch/energy losses are stable or improving, not drifting upward for 3+ epochs
- [ ] **Stop token behavior:** stop loss remains low and stable (no sudden spikes)
- [ ] **Checkpoint integrity:** checkpoint saved and restorable

---

## 2) Validation & Sample Quality (every 1–2 epochs)

- [ ] Run the same fixed validation subset each time
- [ ] Track validation mel (or your primary validation objective)
- [ ] Generate the same fixed sample prompts each check
- [ ] Listen for: pronunciation errors, skipped phones, repeated phones, trailing noise, early/late stop
- [ ] Compare against previous checkpoint before deciding to continue

### Quick Quality Rubric (subjective)

Score each 1–5:

- [ ] Intelligibility
- [ ] Prosody naturalness
- [ ] Pronunciation consistency
- [ ] End-of-utterance stopping behavior
- [ ] Background artifacts/noise

If total score is not improving by epoch 5–10, investigate before long training.

---

## 3) Decision Gates

## Gate A — End of Epoch 5

Continue if all are true:

- [ ] Training is stable (no NaN/Inf/OOM loops)
- [ ] Validation objective improved vs first measured point
- [ ] Sample quality improved at least modestly
- [ ] No major regression in pronunciation/stop behavior

If not true, tune first (batch size, LR schedule bounds, data filtering, duration alignment sanity).

## Gate B — End of Epoch 10

Continue long run if all are true:

- [ ] Clear downward trend in validation objective
- [ ] Audible quality improved from epoch 5 samples
- [ ] No persistent artifacts (buzz, truncation, repeated phones)
- [ ] Throughput/runtime is acceptable for full 100-epoch plan

If not true, stop and retune instead of burning full runtime.

---

## 4) When to Stop Immediately

- [ ] Loss becomes NaN/Inf and repeats after restart
- [ ] Frequent OOM despite conservative settings
- [ ] Validation consistently worsens for 3+ checks
- [ ] Audible quality regresses for multiple checkpoints

---

## 5) Minimal Tracking Table (copy per epoch)

| Epoch | Train total | Train mel | Val mel/objective | LR (end) | Mem state | Sample score (/25) | Decision |
|---|---:|---:|---:|---:|---|---:|---|
| 2 |  |  |  |  |  |  |  |
| 3 |  |  |  |  |  |  |  |
| 4 |  |  |  |  |  |  |  |
| 5 (Gate A) |  |  |  |  |  |  | Continue / Tune |
| 6 |  |  |  |  |  |  |  |
| 7 |  |  |  |  |  |  |  |
| 8 |  |  |  |  |  |  |  |
| 9 |  |  |  |  |  |  |  |
| 10 (Gate B) |  |  |  |  |  |  | Continue / Stop |
