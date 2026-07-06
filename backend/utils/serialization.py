"""JSON helpers that tolerate numpy scalars / arrays / tuples-as-keys."""
from __future__ import annotations

import json
import math
from pathlib import Path


def json_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=json_default)


def load_json(path: Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_jsonable(payload):
    """Round-trip through json to strip numpy types (for API responses)."""
    return json.loads(json.dumps(payload, default=json_default))
