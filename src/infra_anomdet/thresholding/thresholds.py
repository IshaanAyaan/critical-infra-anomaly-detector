
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from scipy.stats import genpareto

def quantile_threshold(scores: np.ndarray, q: float) -> float:
    return float(np.quantile(scores, q))

def pot_threshold(scores: np.ndarray, base_quantile: float = 0.98, target_fpr: float = 1e-3) -> Dict[str, float]:
    # Fit GPD to excesses over high threshold u (EVT POT).
    # We assume scores are from normal/SAFE validation data.
    scores = np.asarray(scores, dtype=float)
    u = np.quantile(scores, base_quantile)
    excess = scores[scores > u] - u
    if excess.size < 50:
        # not enough tail samples -> fallback to high quantile
        thr = float(np.quantile(scores, 1.0 - target_fpr))
        return {"type": "quantile_fallback", "u": float(u), "threshold": thr, "base_quantile": float(base_quantile), "target_fpr": float(target_fpr)}
    c, loc, scale = genpareto.fit(excess, floc=0.0)
    # P(X>t) = p_u * (1 + c*(t-u)/scale)^(-1/c), where p_u = P(X>u)
    p_u = float((scores > u).mean())
    # want P(X>t) = target_fpr
    # target_fpr = p_u * (1 + c*(t-u)/scale)^(-1/c)
    # (target_fpr/p_u)^(-c) = 1 + c*(t-u)/scale
    ratio = max(target_fpr / max(p_u, 1e-12), 1e-12)
    t = u + (scale / c) * (ratio ** (-c) - 1.0) if abs(c) > 1e-8 else u + scale * np.log(1.0/ratio)
    return {"type": "pot", "u": float(u), "threshold": float(t), "c": float(c), "scale": float(scale), "base_quantile": float(base_quantile), "target_fpr": float(target_fpr), "p_u": float(p_u)}
