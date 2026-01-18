
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

def save_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def robust_scale_fit(X: np.ndarray, eps: float = 1e-9) -> Dict[str, Any]:
    # X: (N, F) or (N, T, F)
    X2 = X.reshape(-1, X.shape[-1])
    med = np.median(X2, axis=0)
    q1 = np.percentile(X2, 25, axis=0)
    q3 = np.percentile(X2, 75, axis=0)
    iqr = np.maximum(q3 - q1, eps)
    return {"median": med.tolist(), "iqr": iqr.tolist()}

def robust_scale_transform(X: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    med = np.array(scaler["median"], dtype=np.float32)
    iqr = np.array(scaler["iqr"], dtype=np.float32)
    return (X - med) / iqr

def sliding_windows(X: np.ndarray, y: Optional[np.ndarray], seq_len: int, stride: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # X: (N, F), y: (N,) optional
    n = X.shape[0]
    idxs = list(range(0, n - seq_len + 1, stride))
    W = np.stack([X[i:i+seq_len] for i in idxs], axis=0).astype(np.float32)  # (B, T, F)
    if y is None:
        return W, None
    Wy = np.array([y[i:i+seq_len].max() for i in idxs], dtype=np.int64)
    return W, Wy

def engineered_features(W: np.ndarray) -> np.ndarray:
    # W: (B, T, F) -> (B, 6F)
    # mean, std, min, max, last-first, mean abs diff
    mean = W.mean(axis=1)
    std = W.std(axis=1)
    mn = W.min(axis=1)
    mx = W.max(axis=1)
    delta = W[:, -1, :] - W[:, 0, :]
    mad = np.abs(np.diff(W, axis=1)).mean(axis=1)
    return np.concatenate([mean, std, mn, mx, delta, mad], axis=1)
