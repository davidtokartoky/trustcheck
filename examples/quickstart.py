# -*- coding: utf-8 -*-
"""
trustcheck quickstart — a real (tiny) training run that lies, and trustcheck
catching it. No GPU, no downloads, runs in seconds. `python examples/quickstart.py`

We deliberately train a model to MEMORIZE (tiny dataset, big model, no reg) so
you can see trustcheck flag it — the same failure that fooled us for real.
"""

import random
from trustcheck import TrustMonitor

# --- a toy "model": memorizes a lookup table, generalizes poorly ------------
# We simulate a training loop where train accuracy climbs to ~1.0 while held-out
# accuracy stalls. This is exactly the shape of a memorizing fine-tune.
random.seed(0)


def simulate_epoch(epoch):
    # train acc rushes toward 1.0
    train = min(1.0, 0.4 + 0.13 * epoch + random.uniform(-0.02, 0.02))
    # eval acc stalls well below — memorization, not learning
    eval_ = min(0.62, 0.35 + 0.06 * epoch + random.uniform(-0.03, 0.03))
    return round(train, 3), round(eval_, 3)


def main():
    mon = TrustMonitor(higher_is_better=True)

    print("epoch | train_acc | eval_acc")
    print("-" * 32)
    for epoch in range(6):
        train_acc, eval_acc = simulate_epoch(epoch)
        print(f"  {epoch}   |   {train_acc:.3f}   |  {eval_acc:.3f}")
        # >>> the one line you add to your own loop: <<<
        mon.record(step=epoch, eval_metric=eval_acc, train_metric=train_acc)

    print("\nWithout trustcheck you'd screenshot the last eval number and move on.")
    print("With it:\n")
    print(mon.verdict())


if __name__ == "__main__":
    main()
