
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

def pointwise_metrics(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    out = {}
    try:
        out["auroc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y_true, scores))
    except Exception:
        out["auprc"] = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred.astype(int), average="binary", zero_division=0)
    out["precision"] = float(p)
    out["recall"] = float(r)
    out["f1"] = float(f1)
    out["anomaly_rate_true"] = float(y_true.mean())
    out["anomaly_rate_pred"] = float(y_pred.mean())
    return out

def events_from_labels(y: np.ndarray) -> List[Tuple[int,int]]:
    # contiguous segments of y==1 as events
    y = y.astype(int)
    ev = []
    i = 0
    n = len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            ev.append((i, j-1))
            i = j
        else:
            i += 1
    return ev

def event_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_events = events_from_labels(y_true)
    pred_events = events_from_labels(y_pred)
    if len(true_events) == 0:
        return {"event_recall": float("nan"), "false_incidents": float(len(pred_events)), "mean_delay": float("nan")}
    # event recall: a true event is detected if any predicted anomaly overlaps it
    hits = 0
    delays = []
    for s,e in true_events:
        detected = False
        first = None
        for ps,pe in pred_events:
            if pe < s: 
                continue
            if ps > e:
                break
            # overlap
            detected = True
            first = max(ps, s)
            break
        if detected:
            hits += 1
            delays.append(first - s)
    recall = hits / len(true_events)
    false_incidents = 0
    for ps,pe in pred_events:
        overlaps = any(not (pe < s or ps > e) for s,e in true_events)
        if not overlaps:
            false_incidents += 1
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    return {"event_recall": float(recall), "false_incidents": float(false_incidents), "mean_delay": mean_delay}
