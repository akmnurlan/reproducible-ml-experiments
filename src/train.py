from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data import DataConfig, make_synthetic_classification
from src.eval import aggregate_folds, evaluate_fold
from src.model import ModelConfig, build_model
from src.utils.seed import set_seed
from src.utils.logging import append_run_row, save_config_snapshot


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_sweeps(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand simple sweeps under:
      model:
        sweep:
          C: [0.1, 1.0]
    Returns list of concrete configs.
    """
    model = cfg.get("model", {})
    sweep = model.get("sweep")
    if not sweep:
        return [cfg]

    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    expanded = []

    for combo in product(*values):
        new_cfg = yaml.safe_load(yaml.dump(cfg))  # deep copy via YAML round-trip
        for k, v in zip(keys, combo):
            new_cfg["model"].setdefault("params", {})
            new_cfg["model"]["params"][k] = v
        # Remove sweep from final snapshot to keep it clean
        new_cfg["model"].pop("sweep", None)
        expanded.append(new_cfg)

    return expanded


def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_seed = int(cfg["run"]["seed"])
    set_seed(run_seed)

    output_dir = cfg["experiment"]["output_dir"]
    cfg_id = save_config_snapshot(output_dir, cfg)

    # Build data + model configs
    dcfg = DataConfig(**cfg["data"])
    X, y = make_synthetic_classification(dcfg, seed=run_seed)

    mcfg = ModelConfig(type=cfg["model"]["type"], params=cfg["model"].get("params", {}))

    cv_cfg = cfg["cv"]
    folds = int(cv_cfg.get("folds", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))

    skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=run_seed if shuffle else None)

    fold_metrics: List[Dict[str, float]] = []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = build_model(mcfg, seed=run_seed)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        metrics = evaluate_fold(y[te], pred)
        fold_metrics.append(metrics)

    agg = aggregate_folds(fold_metrics)

    row = {
        "experiment": cfg["experiment"]["name"],
        "config_id": cfg_id,
        "model": mcfg.type,
        "seed": run_seed,
        "folds": folds,
        **agg,
    }
    append_run_row(output_dir, row)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    expanded = expand_sweeps(base_cfg)

    print(f"Loaded {args.config}. Running {len(expanded)} configuration(s).")

    for i, cfg in enumerate(expanded, start=1):
        row = run_one(cfg)
        print(f"[{i}/{len(expanded)}] {row}")


if __name__ == "__main__":
    main()
