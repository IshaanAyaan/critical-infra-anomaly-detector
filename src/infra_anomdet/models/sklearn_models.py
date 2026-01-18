
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib

@dataclass
class SklearnDetector:
    kind: str
    model: Any

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        # larger = more anomalous
        if hasattr(self.model, "decision_function"):
            return -self.model.decision_function(X)
        if hasattr(self.model, "score_samples"):
            return -self.model.score_samples(X)
        raise ValueError("Model does not support scoring")

    def save(self, path: str) -> None:
        joblib.dump({"kind": self.kind, "model": self.model}, path)

    @staticmethod
    def load(path: str) -> "SklearnDetector":
        obj = joblib.load(path)
        return SklearnDetector(kind=obj["kind"], model=obj["model"])

def make_iforest(n_estimators: int = 300, contamination: float = 0.001, random_state: int = 42) -> SklearnDetector:
    m = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state, n_jobs=-1)
    return SklearnDetector(kind="iforest", model=m)

def make_ocsvm(nu: float = 0.01, kernel: str = "rbf", gamma: str = "scale") -> SklearnDetector:
    m = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    return SklearnDetector(kind="ocsvm", model=m)
