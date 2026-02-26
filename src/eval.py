from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score


@dataclass(frozen=True)
class FoldResult:
    fold: int
    acc: float


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def aggregate_folds(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    accs = [m["accuracy"] for m in fold_metrics]
    return {
        "cv_accuracy_mean": float(np.mean(accs)),
        "cv_accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
    }
