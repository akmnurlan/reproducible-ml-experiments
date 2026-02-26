from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import hashlib


def _stable_hash_dict(d: Dict[str, Any]) -> str:
    """
    Create a stable hash for a nested dict (config snapshot).
    """
    blob = json.dumps(d, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def ensure_results_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_config_snapshot(output_dir: str, config: Dict[str, Any]) -> str:
    """
    Save a config snapshot into results/ and return the config hash id.
    """
    out = ensure_results_dir(output_dir)
    cfg_id = _stable_hash_dict(config)
    snap_path = out / f"config_{cfg_id}.json"
    if not snap_path.exists():
        snap_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    return cfg_id


def append_run_row(
    output_dir: str,
    row: Dict[str, Any],
    filename: str = "runs.csv",
) -> None:
    """
    Append one experiment row to results/runs.csv with a header (created if missing).
    """
    out = ensure_results_dir(output_dir)
    path = out / filename

    # Always add timestamp (UTC ISO)
    row = dict(row)
    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    fieldnames = list(row.keys())

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
