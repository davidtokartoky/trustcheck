# -*- coding: utf-8 -*-
"""
Tests for trustcheck. Run with: pytest

The whole point of trustcheck is "can you trust this number?" — so it had
better be able to prove it trusts the right ones and distrusts the wrong ones.
Each test encodes one real failure shape we hit during development.
"""

from trustcheck import TrustMonitor, Verdict


# ---------------------------------------------------------------------------
# Healthy run -> TRUST
# ---------------------------------------------------------------------------
def test_healthy_run_is_trusted():
    m = TrustMonitor(higher_is_better=True)
    for step, tr, ev in [(0, .30, .28), (1, .60, .55), (2, .80, .74),
                         (3, .90, .86), (4, .93, .90), (5, .94, .91)]:
        m.record(step, ev, tr)
    v = m.verdict()
    assert v.level == "TRUST"
    assert v.expected_production is None  # no downgrade needed


# ---------------------------------------------------------------------------
# "Learned it then forgot it" -> DO_NOT_TRUST (this is our Step 0 run)
# ---------------------------------------------------------------------------
def test_forgotten_best_checkpoint_is_flagged():
    m = TrustMonitor(higher_is_better=True)
    for step, acc in [(0, .22), (150, .55), (300, .55), (450, 1.00),
                      (600, .83), (750, .61), (899, .44)]:
        m.record(step, acc, min(1.0, acc + .05))
    v = m.verdict()
    assert v.level == "DO_NOT_TRUST"
    assert v.best_metric > v.final_metric
    assert v.best_step == 450
    # the report should tell you to keep the better checkpoint
    assert any("checkpoint" in a.lower() for a in v.actions)


# ---------------------------------------------------------------------------
# Memorization -> DO_NOT_TRUST + honest production estimate
# ---------------------------------------------------------------------------
def test_memorization_is_flagged_with_honest_estimate():
    m = TrustMonitor(higher_is_better=True, memorization_gap=0.10)
    for step, tr, ev in [(0, .40, .35), (1, .70, .50), (2, .90, .58),
                         (3, .99, .61), (4, 1.00, .60)]:
        m.record(step, ev, tr)
    v = m.verdict()
    assert v.level == "DO_NOT_TRUST"
    # honest estimate should be the eval number, not the train number
    assert v.expected_production is not None
    assert abs(v.expected_production - 0.60) < 1e-6


# ---------------------------------------------------------------------------
# Oscillation -> at least CAUTION, warns about seeds
# ---------------------------------------------------------------------------
def test_oscillation_warns_about_seeds():
    m = TrustMonitor(higher_is_better=True, max_drops_for_stable=1)
    # bounces up and down repeatedly but ends near its best
    for step, ev in [(0, .5), (1, .7), (2, .6), (3, .75), (4, .65), (5, .74)]:
        m.record(step, ev, ev + .03)
    v = m.verdict()
    assert v.level in ("CAUTION", "DO_NOT_TRUST")
    assert any("seed" in a.lower() for a in v.actions)


# ---------------------------------------------------------------------------
# Loss mode (lower is better) works symmetrically
# ---------------------------------------------------------------------------
def test_lower_is_better_loss_mode():
    m = TrustMonitor(higher_is_better=False)
    # loss goes down nicely then blows up at the end
    for step, ev in [(0, 2.0), (1, 1.2), (2, 0.8), (3, 0.5), (4, 1.9)]:
        m.record(step, ev)
    v = m.verdict()
    assert v.level == "DO_NOT_TRUST"          # kept a much worse final
    assert v.best_metric < v.final_metric      # lower loss = better


# ---------------------------------------------------------------------------
# Not enough data -> CAUTION, never a false TRUST
# ---------------------------------------------------------------------------
def test_single_point_is_cautious_not_trusted():
    m = TrustMonitor()
    m.record(0, 0.9)
    v = m.verdict()
    assert v.level == "CAUTION"


# ---------------------------------------------------------------------------
# Missing train_metric -> memorization check skipped, still usable
# ---------------------------------------------------------------------------
def test_works_without_train_metric():
    m = TrustMonitor(higher_is_better=True)
    for step, ev in [(0, .5), (1, .7), (2, .85), (3, .88)]:
        m.record(step, ev)  # no train metric
    v = m.verdict()
    assert v.level == "TRUST"
    assert any("memorization" in f.lower() and "skipped" in f.lower()
               for f in v.findings)


# ---------------------------------------------------------------------------
# Fast train convergence (common with LoRA) is NOT memorization if eval
# keeps climbing normally — this is the exact case a reviewer flagged as a
# false-positive risk in the old static-threshold version.
# ---------------------------------------------------------------------------
def test_fast_convergence_is_not_flagged_as_memorization():
    m = TrustMonitor(higher_is_better=True)
    # train rockets to 99%, but eval is climbing steadily too — not stagnant
    for step, tr, ev in [(0, .50, .40), (1, .85, .50), (2, .97, .58),
                         (3, .99, .66), (4, .99, .73)]:
        m.record(step, ev, tr)
    v = m.verdict()
    assert v.level == "TRUST"
    mem = m.memorization()
    assert mem["memorizing"] is False
    assert any("fast convergence" in f.lower() for f in v.findings)


# ---------------------------------------------------------------------------
# Real memorization (gap widens AND eval stagnates) is still caught
# ---------------------------------------------------------------------------
def test_widening_gap_with_stagnant_eval_is_still_memorization():
    m = TrustMonitor(higher_is_better=True)
    for step, tr, ev in [(0, .40, .35), (1, .70, .50), (2, .90, .58),
                         (3, .99, .60), (4, 1.00, .59)]:
        m.record(step, ev, tr)
    v = m.verdict()
    assert v.level == "DO_NOT_TRUST"
    mem = m.memorization()
    assert mem["memorizing"] is True
    assert mem["eval_stagnant"] or mem["gap_widening"]



def test_verdict_is_machine_and_human_readable():
    m = TrustMonitor()
    m.record(0, .5, .5)
    m.record(1, .9, .92)
    v = m.verdict()
    assert isinstance(v, Verdict)
    assert v.level in ("TRUST", "CAUTION", "DO_NOT_TRUST")  # machine
    assert "VERDICT" in str(v)                               # human
