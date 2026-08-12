# -*- coding: utf-8 -*-
"""
trustcheck — Should you trust your fine-tune's validation number?

A tiny, local, dependency-free sanity layer for people training small models.
It watches your training run and tells you, in plain language, whether the
final metric is trustworthy — or whether it's lying to you.

What it catches (v0.1):
  1. INSTABILITY  — your model hit 96% at epoch 7 and you saved the 94% one.
                    ("learned it, then forgot it" — the final number hides this)
  2. MEMORIZATION — train 99% / eval 71% AND eval has stagnated means expect
                    ~71% in production. (Fast train convergence with eval
                    still climbing is NOT flagged — that's normal, not
                    memorization.)
  3. UNSTABLE VERDICT — if the run oscillated, one run is an anecdote;
                    the tool tells you when a seed-check is actually warranted.

What it deliberately does NOT do (yet): leak probes (task-specific),
multi-seed orchestration (expensive — we tell you WHEN you need it instead),
dashboards, cloud anything. Everything runs locally. Your data never leaves.

Usage A — HuggingFace Trainer (zero friction):

    from trustcheck import TrustCheckCallback
    trainer = Trainer(..., callbacks=[TrustCheckCallback()])
    trainer.train()   # verdict prints at the end of training

Usage B — any training loop (framework-agnostic):

    from trustcheck import TrustMonitor
    mon = TrustMonitor()
    for epoch in range(epochs):
        ...train...
        mon.record(step=epoch, eval_metric=val_acc, train_metric=train_acc)
    print(mon.verdict())

MIT license. Single file. No telemetry. If it saved you a bad deploy,
that's the whole point.
"""

from dataclasses import dataclass, field
from typing import List, Optional

__version__ = "0.1.0"
__all__ = ["TrustMonitor", "TrustCheckCallback", "Verdict"]


# ---------------------------------------------------------------------------
# Core data
# ---------------------------------------------------------------------------
@dataclass
class _Point:
    step: float
    eval_metric: float
    train_metric: Optional[float] = None


@dataclass
class Verdict:
    """Machine-readable verdict. str(verdict) gives the human report."""
    level: str                    # "TRUST" | "CAUTION" | "DO_NOT_TRUST"
    final_metric: float
    best_metric: float
    best_step: float
    expected_production: Optional[float]   # honest estimate, None if unknown
    findings: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        icon = {"TRUST": "[OK]", "CAUTION": "[!]", "DO_NOT_TRUST": "[X]"}[self.level]
        lines = []
        lines.append("=" * 62)
        lines.append(f"{icon} TRUSTCHECK VERDICT: {self.level.replace('_', ' ')}")
        lines.append("=" * 62)
        lines.append(f"  final metric: {self.final_metric:.4f}"
                     f"   best: {self.best_metric:.4f} (step {self.best_step:g})")
        if (self.expected_production is not None
                and abs(self.expected_production - self.final_metric) > 0.005):
            lines.append(f"  expect in production: ~{self.expected_production:.4f}"
                         f"  (NOT {self.final_metric:.4f})")
        if self.findings:
            lines.append("-" * 62)
            for f in self.findings:
                lines.append(f"  - {f}")
        if self.actions:
            lines.append("-" * 62)
            lines.append("  what to do:")
            for a in self.actions:
                lines.append(f"    -> {a}")
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The monitor (framework-agnostic core)
# ---------------------------------------------------------------------------
class TrustMonitor:
    """
    Records (step, eval_metric[, train_metric]) points during training and
    produces a plain-language Verdict about whether the final number can be
    trusted.

    Thresholds are deliberately simple and visible — you're supposed to read
    them, disagree, and tune them. No black box.
    """

    def __init__(
        self,
        higher_is_better: bool = True,
        # a "drop" = eval metric got worse by more than this between evals
        drop_tolerance: float = 0.01,
        # best - final larger than this => you kept a worse checkpoint
        forgotten_gap: float = 0.02,
        # train - eval larger than this => memorization warning
        memorization_gap: float = 0.10,
        # more drops than this => run counts as oscillating.
        # 1 means "2+ meaningful drops = oscillating" — deliberately strict:
        # a run that lost ground twice is not something to trust blindly.
        max_drops_for_stable: int = 1,
    ):
        self.higher_is_better = higher_is_better
        self.drop_tolerance = drop_tolerance
        self.forgotten_gap = forgotten_gap
        self.memorization_gap = memorization_gap
        self.max_drops_for_stable = max_drops_for_stable
        self._points: List[_Point] = []

    # -- recording ----------------------------------------------------------
    def record(self, step: float, eval_metric: float,
               train_metric: Optional[float] = None) -> None:
        """Call once per evaluation. train_metric is optional but unlocks
        the memorization check — pass it if you have it."""
        self._points.append(_Point(step, float(eval_metric),
                                   None if train_metric is None
                                   else float(train_metric)))

    # -- internals ----------------------------------------------------------
    def _better(self, a: float, b: float) -> bool:
        return a > b if self.higher_is_better else a < b

    def _worse_by(self, now: float, prev: float) -> float:
        return (prev - now) if self.higher_is_better else (now - prev)

    # -- analysis -----------------------------------------------------------
    def stability(self) -> dict:
        pts = self._points
        drops = 0
        for prev, now in zip(pts, pts[1:]):
            if self._worse_by(now.eval_metric, prev.eval_metric) > self.drop_tolerance:
                drops += 1
        best = pts[0]
        for p in pts[1:]:
            if self._better(p.eval_metric, best.eval_metric):
                best = p
        final = pts[-1]
        forgotten = self._worse_by(final.eval_metric, best.eval_metric)
        return {
            "drops": drops,
            "oscillating": drops > self.max_drops_for_stable,
            "best": best.eval_metric, "best_step": best.step,
            "final": final.eval_metric,
            "forgot": forgotten > self.forgotten_gap,
            "forgotten_by": max(0.0, forgotten),
        }

    def memorization(self) -> Optional[dict]:
        """
        Adaptive memorization check.

        A large train/eval gap ALONE is not enough — with LoRA and similar
        techniques, train accuracy routinely rockets to ~99% while eval is
        still climbing normally. That's fast convergence, not memorization,
        and flagging it would make the tool cry wolf until people stop
        trusting it.

        Real memorization has a second signature: eval STAGNATES (or the
        gap WIDENS) while train stays high. We require BOTH signals —
        a wide gap AND (eval flat-or-falling recently, OR the gap has been
        growing) — before calling it memorization.

        With only 2 points we can't see a trend, so we fall back to the
        simple gap check but say so explicitly in the finding.
        """
        pts = [p for p in self._points if p.train_metric is not None]
        if not pts:
            return None
        last = pts[-1]
        gap = (last.train_metric - last.eval_metric) if self.higher_is_better \
            else (last.eval_metric - last.train_metric)

        if len(pts) < 3:
            return {
                "train": last.train_metric, "eval": last.eval_metric,
                "gap": gap, "memorizing": gap > self.memorization_gap,
                "trend_based": False,
            }

        # look at the second half of recorded points for a trend
        mid = len(pts) // 2
        earlier, recent = pts[:mid + 1], pts[mid:]

        def _gap_at(p):
            return (p.train_metric - p.eval_metric) if self.higher_is_better \
                else (p.eval_metric - p.train_metric)

        gap_earlier = _gap_at(earlier[-1])
        gap_widening = (gap - gap_earlier) > (self.memorization_gap * 0.3)

        eval_recent_change = self._worse_by(recent[-1].eval_metric,
                                            recent[0].eval_metric)
        # negative "worse_by" means eval improved; stagnation = improved
        # by less than a small slice of the gap threshold, or got worse
        eval_stagnant = eval_recent_change > -(self.memorization_gap * 0.2)

        memorizing = gap > self.memorization_gap and (gap_widening or eval_stagnant)

        return {
            "train": last.train_metric, "eval": last.eval_metric,
            "gap": gap, "memorizing": memorizing, "trend_based": True,
            "gap_widening": gap_widening, "eval_stagnant": eval_stagnant,
        }

    # -- verdict ------------------------------------------------------------
    def verdict(self) -> Verdict:
        if len(self._points) < 2:
            return Verdict(
                level="CAUTION",
                final_metric=self._points[-1].eval_metric if self._points else float("nan"),
                best_metric=self._points[-1].eval_metric if self._points else float("nan"),
                best_step=self._points[-1].step if self._points else -1,
                expected_production=None,
                findings=["Fewer than 2 evaluation points recorded — "
                          "nothing meaningful can be checked."],
                actions=["Evaluate at least every epoch so the run's shape is visible."],
            )

        st = self.stability()
        mem = self.memorization()
        findings: List[str] = []
        actions: List[str] = []
        level = "TRUST"
        expected = None

        # 1) "learned it, then forgot it"
        if st["forgot"]:
            level = "DO_NOT_TRUST"
            findings.append(
                f"[stability] Best was {st['best']:.4f} at step {st['best_step']:g}; "
                f"you ended at {st['final']:.4f}. The model learned more than the "
                f"final checkpoint shows — you kept a worse one."
            )
            actions.append(
                f"Re-run with checkpoint saving and keep the step-{st['best_step']:g} "
                f"weights (or use load_best_model_at_end=True)."
            )

        # 2) oscillation => one run is an anecdote
        if st["oscillating"]:
            if level == "TRUST":
                level = "CAUTION"
            findings.append(
                f"[stability] Eval metric dropped {st['drops']}x during training — "
                f"this run OSCILLATES. A single run's final number is an anecdote, "
                f"not a measurement."
            )
            actions.append(
                "Repeat with 2-3 different seeds before believing any number; "
                "consider lower LR / gradient clipping."
            )

        # 3) memorization
        if mem and mem["memorizing"]:
            level = "DO_NOT_TRUST"
            expected = mem["eval"]
            reason = []
            if mem.get("trend_based"):
                if mem.get("gap_widening"):
                    reason.append("the gap is WIDENING over training")
                if mem.get("eval_stagnant"):
                    reason.append("eval has STAGNATED while train stayed high")
                reason_str = " and ".join(reason) if reason else "gap is large"
            else:
                reason_str = "gap is large (too few points to check the trend)"
            findings.append(
                f"[memorization] train {mem['train']:.4f} vs eval {mem['eval']:.4f} "
                f"(gap {mem['gap']:.4f}) — {reason_str}. The model looks like it's "
                f"memorizing training data, not learning the task."
            )
            actions.append(
                f"Treat ~{mem['eval']:.4f} as the honest performance estimate. "
                f"More/more-diverse data or stronger regularization will help; "
                f"a higher train number will not."
            )
        elif mem and mem["gap"] > self.memorization_gap:
            # large gap but neither widening nor stagnant => likely just fast
            # convergence on the training objective (common with LoRA etc.),
            # not memorization. Say so instead of crying wolf.
            findings.append(
                f"[memorization] train {mem['train']:.4f} vs eval {mem['eval']:.4f} "
                f"is a wide gap, but eval is still improving and the gap isn't "
                f"widening — looks like fast convergence on train, not "
                f"memorization. Keep an eye on it if eval plateaus."
            )
        elif mem is None:
            findings.append(
                "[memorization] No train_metric provided — memorization check "
                "skipped. Pass train_metric to record() to enable it."
            )

        if level == "TRUST":
            findings.append(
                "No red flags: run was stable, best ~= final"
                + (", train/eval gap is small." if mem else ".")
            )

        return Verdict(
            level=level,
            final_metric=st["final"],
            best_metric=st["best"],
            best_step=st["best_step"],
            expected_production=expected,
            findings=findings,
            actions=actions,
        )


# ---------------------------------------------------------------------------
# HuggingFace Trainer integration (optional — only loads if transformers exists)
# ---------------------------------------------------------------------------
try:
    from transformers import TrainerCallback  # type: ignore

    class TrustCheckCallback(TrainerCallback):
        """
        Drop-in callback for transformers.Trainer:

            trainer = Trainer(..., callbacks=[TrustCheckCallback()])

        Reads eval metrics from the Trainer's own evaluation loop
        (metric_key, default "eval_loss" -> lower is better; pass e.g.
        metric_key="eval_accuracy", higher_is_better=True for accuracy).
        Prints the verdict at the end of training.
        """

        def __init__(self, metric_key: str = "eval_loss",
                     higher_is_better: Optional[bool] = None,
                     train_key: str = "loss", **monitor_kwargs):
            hib = (not metric_key.endswith("loss")) if higher_is_better is None \
                else higher_is_better
            self.metric_key = metric_key
            self.train_key = train_key
            self.monitor = TrustMonitor(higher_is_better=hib, **monitor_kwargs)
            self._last_train_metric: Optional[float] = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and self.train_key in logs and "eval" not in " ".join(logs):
                self._last_train_metric = logs[self.train_key]

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and self.metric_key in metrics:
                self.monitor.record(
                    step=state.global_step,
                    eval_metric=metrics[self.metric_key],
                    train_metric=self._last_train_metric,
                )

        def on_train_end(self, args, state, control, **kwargs):
            print()
            print(self.monitor.verdict())

except ImportError:  # transformers not installed — TrustMonitor still works
    TrustCheckCallback = None  # type: ignore


# ---------------------------------------------------------------------------
# Self-demo: run `python trustcheck.py` to see it catch a lying run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Demo 1: the run from our own project (Step 0) — final number lied\n")
    mon = TrustMonitor(higher_is_better=True)
    # real shape from our experiment: 4/18 .. 18/18 at step 450 .. back to 8/18
    for step, acc in [(0, .22), (150, .55), (300, .55), (450, 1.00),
                      (600, .83), (750, .61), (899, .44)]:
        mon.record(step=step, eval_metric=acc, train_metric=min(1.0, acc + .05))
    print(mon.verdict())

    print("\n\nDemo 2: a healthy run for contrast\n")
    mon2 = TrustMonitor(higher_is_better=True)
    for step, tr, ev in [(0, .3, .28), (1, .6, .55), (2, .8, .74),
                         (3, .9, .86), (4, .93, .90), (5, .94, .91)]:
        mon2.record(step=step, eval_metric=ev, train_metric=tr)
    print(mon2.verdict())

    print("\n\nDemo 3: memorization — looks great, isn't\n")
    mon3 = TrustMonitor(higher_is_better=True)
    for step, tr, ev in [(0, .4, .35), (1, .7, .5), (2, .9, .58),
                         (3, .99, .61), (4, 1.0, .60)]:
        mon3.record(step=step, eval_metric=ev, train_metric=tr)
    print(mon3.verdict())
