from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


class EarlyStoppingTracker:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_val_loss = float("inf")
        self.wait_count = 0

    def update(self, val_loss: float) -> tuple[bool, bool, int]:
        """
        Returns:
            improved: val_loss có cải thiện hay không.
            should_stop: có kích hoạt early stopping hay không.
            wait_count: số epoch liên tiếp không cải thiện.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait_count = 0
            return True, False, self.wait_count

        self.wait_count += 1
        should_stop = self.wait_count >= self.patience
        return False, should_stop, self.wait_count
