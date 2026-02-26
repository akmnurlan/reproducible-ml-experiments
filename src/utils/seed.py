from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Optional: if you later add torch, you can extend here safely.
    if deterministic:
        # Placeholder for future deterministic flags (e.g., torch).
        pass
