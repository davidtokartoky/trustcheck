# trustcheck

**Should you trust your fine-tune's validation number?**

A tiny (~300 lines, zero deps) sanity layer for people training small models.
It watches your training run and tells you, in plain language, whether the
final metric is trustworthy — or whether it's lying to you.

```
==============================================================
[X] TRUSTCHECK VERDICT: DO NOT TRUST
==============================================================
  final metric: 0.4400   best: 1.0000 (step 450)
--------------------------------------------------------------
  - [stability] Best was 1.0000 at step 450; you ended at
    0.4400. The model learned more than the final checkpoint
    shows — you kept a worse one.
  - [stability] Eval metric dropped 3x during training — this
    run OSCILLATES. A single run's final number is an anecdote,
    not a measurement.
--------------------------------------------------------------
  what to do:
    -> Re-run with checkpoint saving and keep the step-450
       weights (or use load_best_model_at_end=True).
    -> Repeat with 2-3 seeds before believing any number.
==============================================================
```

That output is from a **real training run** (ours). The final log said 44%.
The model had actually hit 100% mid-training and degraded. Without tracking
the run's *shape*, we almost concluded the architecture didn't work.
It did. The number lied. We built this so it can't happen silently again.

## Why this exists

Every tool will happily plot your loss curve. None of them tell you
**when the number at the end of it shouldn't be believed.** Three ways it
lies, all caught here:

1. **"Learned it, then forgot it"** — best checkpoint was mid-training,
   you kept the worse final one. The final number understates your model.
2. **Memorization** — train 99% / eval 71% *and eval has stagnated* means
   expect ~71% in production. (Fast train convergence with eval still
   climbing — common with LoRA — is deliberately **not** flagged; that's
   normal, not memorization. The check looks at the trend, not just a
   snapshot.)
3. **Oscillation** — if the metric bounced around, one run's final number
   is an anecdote. You need seeds before you need conclusions.

## Install

```bash
pip install trustcheck            # core, zero dependencies
pip install trustcheck[hf]        # + HuggingFace Trainer callback
```

Or just copy `src/trustcheck/__init__.py` into your project — it's one file
with no required dependencies.

## Use

**With HuggingFace Trainer** (zero friction):

```python
from trustcheck import TrustCheckCallback

trainer = Trainer(
    ...,
    callbacks=[TrustCheckCallback(metric_key="eval_accuracy")],
)
trainer.train()   # verdict prints when training ends
```

**With any training loop** (framework-agnostic):

```python
from trustcheck import TrustMonitor

mon = TrustMonitor(higher_is_better=True)
for epoch in range(epochs):
    train_acc, val_acc = train_one_epoch(...)
    mon.record(step=epoch, eval_metric=val_acc, train_metric=train_acc)

print(mon.verdict())          # human report
v = mon.verdict()             # or machine-readable:
if v.level == "DO_NOT_TRUST":
    ...
```

**Runnable example** (a real memorizing run, caught — no GPU, seconds):

```bash
python examples/quickstart.py
```

**Tests** (yes, the trust tool has tests):

```bash
pip install trustcheck[dev] && pytest
```

## What it deliberately does NOT do (yet)

- **Leak probes** — detecting eval leakage properly is task-specific;
  a generic version would give false confidence. Planned, carefully.
- **Multi-seed orchestration** — running seeds costs compute; instead the
  verdict tells you *when* seeds are actually warranted.
- **Dashboards, cloud, telemetry** — everything runs locally, your data
  and weights never leave your machine. This will not change.

## Thresholds are visible on purpose

All cutoffs (`memorization_gap=0.10`, `forgotten_gap=0.02`, ...) are plain
constructor arguments. Read them, disagree, tune them. No black box —
if this tool tells you not to trust something, you can see exactly why.

## Roadmap (not built yet — prioritized by what real usage shows we need)

A few improvements are known and deliberately deferred rather than guessed
at now:

- **Smoother trend detection** — the oscillation/forgetting checks currently
  use a fairly direct point-to-point comparison. An exponential moving
  average (with tolerance) would better distinguish "a single 2% wobble"
  from "steadily declining for the last 10 evals."
- **Early-stopping awareness** — if you're already using an early-stopping
  callback, the final checkpoint is usually at-or-near the best one; the
  monitor should recognize that setup and not warn about something that's
  already handled.
- **Quantified seed guidance** — right now the tool says "this run
  oscillated, check multiple seeds." If you *do* run several seeds, it
  could go further: "variance across 5 runs is 8% — that's above a
  reasonable tolerance, results aren't stable yet."
- **Actionable checkpoint recovery** — `Verdict` already exposes
  `best_step`/`best_metric`; a natural next step is optionally returning or
  saving the best checkpoint directly, not just naming the step in text.

None of this is implemented speculatively — it goes in when a real use
case asks for it. If one of these is the thing blocking you, say so in an
issue and it moves up the list.

## Status

v0.1 — extracted from the diagnostic tooling of a research project where
these exact checks repeatedly saved us from wrong conclusions. Honest
question to you: **does this solve a problem you actually have?**
If yes — or if it's missing the one check you'd need — open an issue.
That decides what gets built next.

MIT license.
