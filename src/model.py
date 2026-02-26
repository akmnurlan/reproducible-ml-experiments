from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelConfig:
    type: str
    params: Dict[str, Any]


def build_model(cfg: ModelConfig, seed: int) -> BaseEstimator:
    """
    Build a sklearn model from config.
    """
    model_type = cfg.type.lower().strip()

    if model_type == "logistic_regression":
        # Ensure deterministic behavior
        params = dict(cfg.params)
        params.setdefault("random_state", seed)
        params.setdefault("solver", "lbfgs")

        # Scaling helps LR in many cases; looks professional and is standard.
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**params)),
            ]
        )

    raise ValueError(f"Unknown model type: {cfg.type}")
