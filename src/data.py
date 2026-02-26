from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.datasets import make_classification


@dataclass(frozen=True)
class DataConfig:
    n_samples: int = 2000
    n_features: int = 20
    n_informative: int = 10
    n_redundant: int = 5
    class_sep: float = 1.0
    flip_y: float = 0.01


def make_synthetic_classification(cfg: DataConfig, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_clusters_per_class=2,
        class_sep=cfg.class_sep,
        flip_y=cfg.flip_y,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64)
