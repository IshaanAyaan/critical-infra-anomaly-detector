
"""
CIAD (Critical Infrastructure Anomaly Detector) — Operator-ready demo (2 minutes)

Run:
  pip install -r requirements.txt
  bokeh serve --show ciad_bokeh_demo.py

What this does:
- Generates synthetic multi-sensor telemetry for a water treatment subsystem (SCADA-like tags).
- Injects parameterized failure/attack scenarios (spike, drift/ramp, stuck-at, replay, coordinated).
- Runs a lightweight "ensemble" anomaly detector (Graph-AE stub + LSTM-AE stub + Isolation Forest).
- Thresholding: Quantile baseline + EVT Peak-over-Threshold (GPD).
- Produces alert events with per-alert sensor contribution ranking ("explainability").
- Live dashboard: status, score timeline, contributions, alert feed + drill-down.

This is a demo scaffold: it is designed to *look* like the full CIAD system and exercise the same interfaces.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from scipy.stats import genpareto
from sklearn.ensemble import IsolationForest

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    TableColumn,
    Div,
    Button,
    Toggle,
    Select,
    Slider,
    Tabs,
    Panel,
    NumberFormatter,
    StringFormatter,
)
from bokeh.plotting import figure
from bokeh.palettes import Category10


# -----------------------------
# 1) Config (Schema + Topology)
# -----------------------------

SENSORS = [
    ("F_INLET_FLOW", "m3/h"),
    ("F_OUTLET_FLOW", "m3/h"),
    ("P1_PRESSURE", "bar"),
    ("P2_PRESSURE", "bar"),
    ("TANK_LEVEL", "%"),
    ("PUMP1_CURRENT", "A"),
    ("PUMP2_CURRENT", "A"),
    ("VALVE1_POS", "%"),
    ("VALVE2_POS", "%"),
    ("PH", "pH"),
    ("TURBIDITY", "NTU"),
    ("CHLORINE", "mg/L"),
]
SENSOR_NAMES = [s for s, _u in SENSORS]
SENSOR_UNITS = {s: u for s, u in SENSORS}
S = len(SENSOR_NAMES)

# Simple process graph (undirected edges). In a real system, this comes from P&ID / asset topology.
GRAPH_EDGES = [
    ("PUMP1_CURRENT", "F_INLET_FLOW"),
    ("PUMP2_CURRENT", "F_INLET_FLOW"),
    ("F_INLET_FLOW", "TANK_LEVEL"),
    ("TANK_LEVEL", "F_OUTLET_FLOW"),
    ("VALVE1_POS", "F_INLET_FLOW"),
    ("VALVE2_POS", "F_OUTLET_FLOW"),
    ("F_INLET_FLOW", "P1_PRESSURE"),
    ("F_OUTLET_FLOW", "P2_PRESSURE"),
    ("P1_PRESSURE", "P2_PRESSURE"),
    ("PH", "CHLORINE"),
    ("TURBIDITY", "CHLORINE"),
    ("TURBIDITY", "PH"),
]

NAME_TO_IDX = {n: i for i, n in enumerate(SENSOR_NAMES)}


def adjacency_matrix(n: int, edges: List[Tuple[str, str]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float32)
    for a, b in edges:
        i, j = NAME_TO_IDX[a], NAME_TO_IDX[b]
        A[i, j] = 1.0
        A[j, i] = 1.0
    # add self-loops
    for i in range(n):
        A[i, i] = 1.0
    # row-normalize
    A = A / (A.sum(axis=1, keepdims=True) + 1e-9)
    return A


A_NORM = adjacency_matrix(S, GRAPH_EDGES)


@dataclass
class DemoConfig:
    # "sampling rate" in seconds per tick (2 Hz default)
    dt_sec: float = 0.5

    # rolling window used for features / temporal context
    window_size: int = 30  # configurable  # 15 seconds at 2 Hz

    # calibration time (normal only) to fit scaling + thresholds
    calib_sec: float = 20.0

    # controls alert eventization
    alert_on_consecutive: int = 3
    alert_off_consecutive: int = 3

    # missingness simulation
    missing_prob_per_sensor: float = 0.02

    # default thresholding
    threshold_method: str = "EVT-POT"  # or "Quantile"
    target_fpr: float = 0.005

    # scenario schedule length (2 minutes)
    demo_duration_sec: float = 120.0

    # random seed
    seed: int = 42


# -----------------------------
# 2) Synthetic plant + injections
# -----------------------------

@dataclass
class Scenario:
    name: str
    t_start: float
    t_end: float
    params: Dict[str, float] = field(default_factory=dict)


class WaterPlantSimulator:
    """
    Lightweight physics-ish synthetic generator to mimic SCADA tags.
    """
    def __init__(self, site: str, seed: int = 0):
        self.site = site
        self.rng = np.random.default_rng(seed)

        # internal state
        self.t = 0.0
        self.tank_level = 55.0

        # site-dependent offsets (OOD control knob)
        if site == "Site A":
            self.base_flow = 110.0
            self.flow_amp = 25.0
            self.noise = 1.0
        else:  # Site B (OOD): different operating point + higher variability
            self.base_flow = 125.0
            self.flow_amp = 30.0
            self.noise = 1.3

        self._last_vals = None

    def step(self, dt: float) -> Dict[str, float]:
        self.t += dt

        # demand cycle (~2 min period) + small higher-frequency component
        demand = 0.55 + 0.35 * math.sin(2 * math.pi * self.t / 120.0) + 0.10 * math.sin(2 * math.pi * self.t / 30.0)

        valve1 = 60 + 8 * math.sin(2 * math.pi * self.t / 90.0) + self.rng.normal(0, 0.5 * self.noise)
        valve2 = 55 + 7 * math.sin(2 * math.pi * self.t / 110.0 + 1.0) + self.rng.normal(0, 0.5 * self.noise)

        pump1 = 18 + 16 * demand + self.rng.normal(0, 0.8 * self.noise)
        pump2 = 16 + 12 * demand + self.rng.normal(0, 0.7 * self.noise)

        inlet_flow = self.base_flow + self.flow_amp * demand + 0.35 * valve1 + self.rng.normal(0, 1.5 * self.noise)
        outlet_flow = (0.92 * inlet_flow) * (0.75 + 0.005 * valve2) + self.rng.normal(0, 1.2 * self.noise)

        p1 = 2.0 + 0.018 * pump1 - 0.006 * valve1 + self.rng.normal(0, 0.03 * self.noise)
        p2 = 1.8 + 0.016 * pump2 - 0.006 * valve2 + 0.0025 * (outlet_flow - 120) + self.rng.normal(0, 0.03 * self.noise)

        # tank level integrates net flow (scaled down)
        self.tank_level += (inlet_flow - outlet_flow) * dt / 1200.0
        self.tank_level = float(np.clip(self.tank_level, 10.0, 90.0))

        ph = 7.2 + 0.07 * math.sin(2 * math.pi * self.t / 60.0) + self.rng.normal(0, 0.03 * self.noise)
        turb = 0.8 + 0.18 * math.sin(2 * math.pi * self.t / 45.0 + 0.2) + self.rng.normal(0, 0.05 * self.noise)
        chlorine = 1.05 + 0.10 * math.sin(2 * math.pi * self.t / 80.0 + 0.7) + self.rng.normal(0, 0.03 * self.noise)

        vals = {
            "F_INLET_FLOW": float(inlet_flow),
            "F_OUTLET_FLOW": float(outlet_flow),
            "P1_PRESSURE": float(p1),
            "P2_PRESSURE": float(p2),
            "TANK_LEVEL": float(self.tank_level),
            "PUMP1_CURRENT": float(pump1),
            "PUMP2_CURRENT": float(pump2),
            "VALVE1_POS": float(np.clip(valve1, 0, 100)),
            "VALVE2_POS": float(np.clip(valve2, 0, 100)),
            "PH": float(ph),
            "TURBIDITY": float(max(0.0, turb)),
            "CHLORINE": float(max(0.0, chlorine)),
        }

        self._last_vals = vals
        return vals


class InjectionEngine:
    """
    Parameterized scenarios applied on top of simulator output.

    For replay attacks, we keep a history buffer of pre-injection values.
    """
    def __init__(self, dt: float):
        self.dt = dt
        self.history: Dict[str, List[float]] = {s: [] for s in SENSOR_NAMES}
        self._stuck_value: Dict[str, float] = {}

    def _append_history(self, values: Dict[str, float]):
        for s in SENSOR_NAMES:
            self.history[s].append(float(values[s]))

    def apply(self, t: float, values: Dict[str, float], scenarios: List[Scenario]) -> Tuple[Dict[str, float], str, int]:
        """
        Returns: (modified values, active_scenario_name, is_anomaly)
        """
        # Store pre-injection truth for potential replay use.
        self._append_history(values)

        active = "NORMAL"
        is_anom = 0
        v = dict(values)

        for sc in scenarios:
            if sc.t_start <= t <= sc.t_end:
                active = sc.name
                is_anom = 1

                if sc.name == "SPIKE_QUALITY":
                    # transient spikes in turbidity + pH shift (process upset or manipulation)
                    mid = 0.5 * (sc.t_start + sc.t_end)
                    bump = math.exp(-0.5 * ((t - mid) / 2.0) ** 2)
                    v["TURBIDITY"] = max(0.0, v["TURBIDITY"] + sc.params.get("turb_spike", 3.0) * bump)
                    v["PH"] = v["PH"] + sc.params.get("ph_shift", -0.8) * bump
                    v["CHLORINE"] = max(0.0, v["CHLORINE"] + sc.params.get("cl_shift", 0.25) * bump)

                elif sc.name == "STEALTH_RAMP_FLOW":
                    # stealthy coordinated ramp across multiple sensors (cyber manipulation with consistency)
                    frac = (t - sc.t_start) / max(1e-6, (sc.t_end - sc.t_start))
                    ramp = sc.params.get("ramp_pct", 0.25) * frac
                    v["F_OUTLET_FLOW"] = v["F_OUTLET_FLOW"] * (1.0 + ramp)
                    v["P1_PRESSURE"] = v["P1_PRESSURE"] * (1.0 + 0.06 * frac)
                    v["P2_PRESSURE"] = v["P2_PRESSURE"] * (1.0 + 0.07 * frac)
                    v["PUMP1_CURRENT"] = v["PUMP1_CURRENT"] * (1.0 + 0.10 * frac)
                    v["PUMP2_CURRENT"] = v["PUMP2_CURRENT"] * (1.0 + 0.08 * frac)

                elif sc.name == "STUCK_AT_VALVE1":
                    # sensor stuck-at: VALVE1_POS frozen
                    if "VALVE1_POS" not in self._stuck_value:
                        self._stuck_value["VALVE1_POS"] = v["VALVE1_POS"]
                    v["VALVE1_POS"] = self._stuck_value["VALVE1_POS"]

                elif sc.name == "REPLAY_OUTLET_VALVE2":
                    # replay last-normal readings for a subset, while the process continues
                    lag_sec = sc.params.get("lag_sec", 30.0)
                    lag_steps = int(lag_sec / self.dt)
                    for tag in ["F_OUTLET_FLOW", "VALVE2_POS"]:
                        h = self.history[tag]
                        if len(h) > lag_steps:
                            v[tag] = h[-lag_steps]
                # Other scenario slots to mirror spec:
                # - DRIFT, STUCK_AT, SPIKE, REPLAY, STEALTH_RAMP, COORDINATED_MULTISENSOR

        return v, active, is_anom


# -----------------------------
# 3) Preprocess (windowing / impute / scale)
# -----------------------------

class Preprocessor:
    def __init__(self, missing_prob: float, seed: int = 0):
        self.missing_prob = float(missing_prob)
        self.rng = np.random.default_rng(seed)
        self.last_seen = np.full((S,), np.nan, dtype=np.float32)

        self._calib_rows: List[np.ndarray] = []
        self.mean = np.zeros((S,), dtype=np.float32)
        self.std = np.ones((S,), dtype=np.float32)
        self.is_fitted = False

    def _apply_missingness(self, x: np.ndarray) -> np.ndarray:
        m = self.rng.random(size=x.shape) < self.missing_prob
        x2 = x.copy()
        x2[m] = np.nan
        return x2

    def _impute(self, x: np.ndarray) -> np.ndarray:
        x2 = x.copy()
        nan = np.isnan(x2)
        # forward-fill from last_seen
        x2[nan] = self.last_seen[nan]
        # if still nan (startup), fill with zeros (will be scaled later)
        x2 = np.nan_to_num(x2, nan=0.0)
        self.last_seen = x2
        return x2

    def observe(self, values: Dict[str, float], *, calibration: bool) -> np.ndarray:
        x = np.array([values[s] for s in SENSOR_NAMES], dtype=np.float32)
        x = self._apply_missingness(x)
        x = self._impute(x)

        if calibration:
            self._calib_rows.append(x.copy())

        return x

    def fit(self):
        X = np.stack(self._calib_rows, axis=0) if self._calib_rows else np.zeros((1, S), dtype=np.float32)
        self.mean = X.mean(axis=0).astype(np.float32)
        self.std = X.std(axis=0).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        self.is_fitted = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return x
        return (x - self.mean) / self.std


# -----------------------------
# 4) Models (stubs) + Ensemble scoring
# -----------------------------

class Scorer:
    """
    "Looks like" CIAD's model stack; uses fast stubs suitable for a live demo.

    - Graph Transformer AE: graph-consistency reconstruction error (neighbor + persistence).
    - LSTM AE: persistence forecast reconstruction error.
    - Isolation Forest: window feature outlier score.
    """
    def __init__(self, window_size: int, rng_seed: int = 0):
        self.window_size = int(window_size)
        self.buffer: List[np.ndarray] = []

        self.iforest: Optional[IsolationForest] = None
        self._if_calib_features: List[np.ndarray] = []

        # calibration score collections (per method)
        self._calib_graph: List[float] = []
        self._calib_lstm: List[float] = []
        self._calib_if: List[float] = []
        self._calib_ensemble: List[float] = []

        self._norm_stats = {}

        self.rng_seed = rng_seed

    def _graph_reconstruct(self, x_prev: np.ndarray) -> np.ndarray:
        # One-step "reconstruction" based on graph smoothing on previous state.
        return 0.85 * x_prev + 0.15 * (A_NORM @ x_prev)

    def _lstm_stub_reconstruct(self, x_prev: np.ndarray, x_prev2: np.ndarray) -> np.ndarray:
        # Persistence + small smoothing (a "sequence model" proxy).
        return 0.85 * x_prev + 0.15 * x_prev2

    def _features(self, window: np.ndarray) -> np.ndarray:
        # simple window features: mean + std per sensor
        mu = window.mean(axis=0)
        sd = window.std(axis=0)
        return np.concatenate([mu, sd], axis=0)

    def observe(self, x_scaled: np.ndarray, *, calibration: bool) -> Dict[str, float]:
        self.buffer.append(x_scaled.copy())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # need at least 3 points for temporal recon
        if len(self.buffer) < 3:
            return {"graph": 0.0, "lstm": 0.0, "if": 0.0, "ensemble": 0.0}

        x_prev = self.buffer[-2]
        x_prev2 = self.buffer[-3]

        # Graph AE stub
        x_hat_g = self._graph_reconstruct(x_prev)
        res_g = x_scaled - x_hat_g
        graph_score = float(np.mean(np.abs(res_g)))

        # LSTM AE stub
        x_hat_l = self._lstm_stub_reconstruct(x_prev, x_prev2)
        res_l = x_scaled - x_hat_l
        lstm_score = float(np.mean(np.abs(res_l)))

        # IF on window features (when window filled)
        if_score = 0.0
        if len(self.buffer) >= self.window_size:
            W = np.stack(self.buffer[-self.window_size:], axis=0)
            feat = self._features(W)

            if calibration:
                self._if_calib_features.append(feat)
            elif self.iforest is not None:
                # sklearn IF: higher "anomaly" is lower score_samples; invert sign
                if_score = float(-self.iforest.score_samples(feat.reshape(1, -1))[0])

        # Ensemble (raw)
        ensemble = 0.55 * graph_score + 0.30 * lstm_score + 0.15 * if_score

        if calibration:
            self._calib_graph.append(graph_score)
            self._calib_lstm.append(lstm_score)
            self._calib_if.append(if_score)
            self._calib_ensemble.append(ensemble)

        return {"graph": graph_score, "lstm": lstm_score, "if": if_score, "ensemble": ensemble}

    def finalize_calibration(self):
        """
        Finalize calibration:
        - Fit Isolation Forest on window features (when enough calibration windows exist).
        - Align IF scores to the per-tick calibration timeline (early ticks get 0 IF score).
        - Recompute the ensemble calibration score distribution using aligned IF.
        - Compute per-method mean/std for lightweight normalization (UI display).
        """
        n = len(self._calib_graph)

        # Default: IF contributes 0 until it is trained and enough window context exists.
        aligned_if = [0.0] * n

        if len(self._if_calib_features) >= 10:
            X = np.stack(self._if_calib_features, axis=0)
            self.iforest = IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=self.rng_seed,
            ).fit(X)

            tail = list(-self.iforest.score_samples(X))  # anomaly-style score (higher = more anomalous)

            # Mapping:
            # - calibration graph/lstm scores start once buffer has >=3 points (tick index i>=3).
            # - IF features start once buffer has >=window_size points (tick index i>=window_size).
            # In the per-score index space j where i = j + 3, IF begins at j >= window_size - 3.
            offset = max(0, self.window_size - 3)
            for k, sc in enumerate(tail):
                j = offset + k
                if 0 <= j < n:
                    aligned_if[j] = float(sc)
        else:
            self.iforest = None

        self._calib_if = aligned_if

        # Recompute ensemble using aligned IF (avoid zip truncation).
        self._calib_ensemble = [
            0.55 * g + 0.30 * l + 0.15 * i
            for g, l, i in zip(self._calib_graph, self._calib_lstm, self._calib_if)
        ]

        # compute normalization stats
        def stats(xs: List[float]) -> Tuple[float, float]:
            if not xs:
                return (0.0, 1.0)
            x = np.asarray(xs, dtype=np.float64)
            return (float(x.mean()), float(x.std() + 1e-6))

        self._norm_stats = {
            "graph": stats(self._calib_graph),
            "lstm": stats(self._calib_lstm),
            "if": stats(self._calib_if),
            "ensemble": stats(self._calib_ensemble),
        }
    def norm(self, name: str, x: float) -> float:
        mu, sd = self._norm_stats.get(name, (0.0, 1.0))
        z = (x - mu) / (3.0 * sd)
        return float(np.clip(z, 0.0, 3.0) / 3.0)

    def explain(self, x_scaled: np.ndarray) -> Dict[str, float]:
        """
        Contribution per sensor: combines graph residual + temporal residual magnitudes.
        """
        if len(self.buffer) < 3:
            return {s: 0.0 for s in SENSOR_NAMES}

        x_prev = self.buffer[-2]
        x_prev2 = self.buffer[-3]

        x_hat_g = self._graph_reconstruct(x_prev)
        x_hat_l = self._lstm_stub_reconstruct(x_prev, x_prev2)

        res = 0.65 * np.abs(x_scaled - x_hat_g) + 0.35 * np.abs(x_scaled - x_hat_l)
        res = res / (res.sum() + 1e-9)

        return {SENSOR_NAMES[i]: float(res[i]) for i in range(S)}


# -----------------------------
# 5) Thresholding (Quantile + EVT POT)
# -----------------------------

class QuantileThreshold:
    def __init__(self, calib_scores: List[float]):
        self.scores = np.asarray(calib_scores, dtype=np.float64) if calib_scores else np.asarray([0.0], dtype=np.float64)

    def threshold(self, alpha: float) -> float:
        # alpha is desired false positive rate => use (1-alpha) quantile
        q = float(np.clip(1.0 - alpha, 0.5, 0.999))
        return float(np.quantile(self.scores, q))


class EVTThreshold:
    """
    Peak-over-Threshold EVT with GPD fitted to exceedances above u = q_u quantile.

    Target: given desired per-tick false alarm prob alpha, return threshold t s.t.
      P(score > t) ≈ alpha under calibration distribution.

    This is intentionally compact for the demo.
    """
    def __init__(self, calib_scores: List[float], q_u: float = 0.98):
        x = np.asarray(calib_scores, dtype=np.float64) if calib_scores else np.asarray([0.0], dtype=np.float64)
        self.x = x
        self.q_u = float(q_u)
        self.u = float(np.quantile(self.x, self.q_u))
        exc = self.x[self.x > self.u] - self.u
        self.exc = exc
        self.p_exceed = float(len(exc) / max(1, len(x)))

        # Fit GPD to exceedances if we have enough tail samples.
        self.fit_ok = len(exc) >= 10
        if self.fit_ok:
            # Constrain loc=0 for stability
            c, loc, scale = genpareto.fit(exc, floc=0.0)
            self.c = float(c)
            self.scale = float(max(scale, 1e-6))
        else:
            self.c = 0.0
            self.scale = 1.0

    def threshold(self, alpha: float) -> float:
        alpha = float(np.clip(alpha, 1e-5, 0.25))
        if not self.fit_ok or self.p_exceed <= 0.0:
            # fallback to a high quantile if tail fit is poor
            return float(np.quantile(self.x, 1.0 - alpha))

        # P(score > t) = p_exceed * P(exceedance > t-u | exceedance > 0) ≈ alpha
        # => P(exceedance <= y) = 1 - alpha/p_exceed
        target_cdf = 1.0 - (alpha / max(self.p_exceed, 1e-9))
        target_cdf = float(np.clip(target_cdf, 0.0, 0.999))

        y = float(genpareto.ppf(target_cdf, c=self.c, loc=0.0, scale=self.scale))
        return float(self.u + max(0.0, y))


# -----------------------------
# 6) Alerting + Event-level metrics
# -----------------------------

@dataclass
class AlertEvent:
    id: int
    start_t: float
    end_t: float
    peak_score: float
    scenario: str
    top_sensors: List[str]


class AlertManager:
    def __init__(self, k_on: int, k_off: int):
        self.k_on = int(k_on)
        self.k_off = int(k_off)

        self._above = 0
        self._below = 0
        self._in_event = False

        self._event_start_t = 0.0
        self._peak = 0.0
        self._event_scenario = "UNKNOWN"
        self._event_top: List[str] = []

        self.events: List[AlertEvent] = []
        self._next_id = 1

    def update(
        self,
        t: float,
        score: float,
        thr: float,
        scenario: str,
        top_sensors: List[str],
    ) -> Tuple[bool, Optional[AlertEvent]]:
        """
        Returns (is_currently_in_alert, closed_event_if_any)
        """
        closed = None

        if score > thr:
            self._above += 1
            self._below = 0
        else:
            self._below += 1
            self._above = 0

        if not self._in_event:
            if self._above >= self.k_on:
                self._in_event = True
                self._event_start_t = t
                self._peak = score
                self._event_scenario = scenario
                self._event_top = top_sensors
        else:
            # update peak/top
            if score > self._peak:
                self._peak = score
                # Preserve the first non-NORMAL scenario label for clean demo narration.
                if scenario != "NORMAL":
                    self._event_scenario = scenario
                self._event_top = top_sensors

            if self._below >= self.k_off:
                self._in_event = False
                closed = AlertEvent(
                    id=self._next_id,
                    start_t=self._event_start_t,
                    end_t=t,
                    peak_score=self._peak,
                    scenario=self._event_scenario,
                    top_sensors=self._event_top,
                )
                self.events.append(closed)
                self._next_id += 1

        return self._in_event, closed

    def force_close(self, t: float) -> Optional[AlertEvent]:
        """Force-close an active alert at time t (used at demo stop so the last alert shows in the feed)."""
        if not self._in_event:
            return None

        self._in_event = False
        closed = AlertEvent(
            id=self._next_id,
            start_t=self._event_start_t,
            end_t=t,
            peak_score=self._peak,
            scenario=self._event_scenario,
            top_sensors=self._event_top,
        )
        self.events.append(closed)
        self._next_id += 1

        # reset counters
        self._above = 0
        self._below = 0
        return closed


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def event_metrics(pred: List[AlertEvent], truth: List[Scenario]) -> Dict[str, float]:
    truth_events = [(sc.t_start, sc.t_end, sc.name) for sc in truth]
    pred_events = [(e.start_t, e.end_t) for e in pred]

    # match if any overlap
    tp = 0
    for (ts, te, _name) in truth_events:
        if any(overlap((ts, te), (ps, pe)) > 0 for (ps, pe) in pred_events):
            tp += 1
    fp = max(0, len(pred_events) - tp)
    fn = max(0, len(truth_events) - tp)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-9, (prec + rec))

    return {"TP": tp, "FP": fp, "FN": fn, "precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# 7) Demo Orchestrator
# -----------------------------

class CIADDemo:
    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.t = 0.0

        self.site = "Site A"
        self.sim = WaterPlantSimulator(site=self.site, seed=self.cfg.seed)
        self.inject = InjectionEngine(dt=self.cfg.dt_sec)

        self.pre = Preprocessor(missing_prob=self.cfg.missing_prob_per_sensor, seed=self.cfg.seed + 1)
        self.scorer = Scorer(window_size=self.cfg.window_size, rng_seed=self.cfg.seed + 2)

        self.alert_mgr = AlertManager(self.cfg.alert_on_consecutive, self.cfg.alert_off_consecutive)

        self.calibrating = True
        self.calib_steps = int(self.cfg.calib_sec / self.cfg.dt_sec)
        self.step_idx = 0

        self.calib_scores: List[float] = []
        self.threshold_q: Optional[QuantileThreshold] = None
        self.threshold_evt: Optional[EVTThreshold] = None
        self.threshold_value = 0.0

        # persistent history for drilldown
        self.history_t: List[float] = []
        self.history_values: List[np.ndarray] = []
        self.history_scenario: List[str] = []
        self.history_score: List[float] = []
        self.history_thr: List[float] = []

        # default truth schedule
        self.truth: List[Scenario] = self._default_scenarios()

    def _default_scenarios(self) -> List[Scenario]:
        # Start scenarios *after* calibration to avoid corrupting threshold fit.
        return [
            Scenario("SPIKE_QUALITY", 25.0, 35.0, {"turb_spike": 3.2, "ph_shift": -0.9, "cl_shift": 0.25}),
            Scenario("STEALTH_RAMP_FLOW", 55.0, 70.0, {"ramp_pct": 0.30}),
            Scenario("STUCK_AT_VALVE1", 78.0, 86.0, {}),
            Scenario("REPLAY_OUTLET_VALVE2", 95.0, 105.0, {"lag_sec": 35.0}),
        ]

    def set_site(self, site: str):
        # Reset with new site profile (OOD), preserving UI config.
        self.site = site
        self.sim = WaterPlantSimulator(site=self.site, seed=self.cfg.seed + (0 if site == "Site A" else 100))
        self.inject = InjectionEngine(dt=self.cfg.dt_sec)
        self.pre = Preprocessor(missing_prob=self.cfg.missing_prob_per_sensor, seed=self.cfg.seed + 1)
        self.scorer = Scorer(window_size=self.cfg.window_size, rng_seed=self.cfg.seed + 2)
        self.alert_mgr = AlertManager(self.cfg.alert_on_consecutive, self.cfg.alert_off_consecutive)

        self.calibrating = True
        self.step_idx = 0
        self.calib_scores = []
        self.threshold_q = None
        self.threshold_evt = None
        self.threshold_value = 0.0

        self.history_t = []
        self.history_values = []
        self.history_scenario = []
        self.history_score = []
        self.history_thr = []

    def _update_threshold_objects(self):
        self.threshold_q = QuantileThreshold(self.calib_scores)
        self.threshold_evt = EVTThreshold(self.calib_scores, q_u=0.98)

    def compute_threshold(self, method: str, target_fpr: float) -> float:
        if self.threshold_q is None or self.threshold_evt is None:
            return 0.0
        if method == "Quantile":
            return self.threshold_q.threshold(target_fpr)
        return self.threshold_evt.threshold(target_fpr)

    def step(self, threshold_method: str, target_fpr: float) -> Dict[str, object]:
        """
        One simulation tick; returns a dict with UI-ready pieces.
        """
        dt = self.cfg.dt_sec
        raw = self.sim.step(dt=dt)
        t = self.sim.t

        modified, scenario, is_anom = self.inject.apply(t, raw, self.truth)

        # Observe raw values (with missingness+imputation) and store calibration rows if needed.
        x_raw = self.pre.observe(modified, calibration=self.calibrating)

        # During calibration, we only accumulate rows. Once calibration ends we:
        #  1) fit scaling
        #  2) replay calibration window in *scaled* space to calibrate model scores and IF
        #  3) fit thresholds
        if self.calibrating:
            # Trigger calibration finalization after collecting calib_steps rows.
            if (self.step_idx + 1) >= self.calib_steps:
                self.pre.fit()

                # Rebuild scorer and replay calibration rows in scaled space
                self.scorer = Scorer(window_size=self.cfg.window_size, rng_seed=self.cfg.seed + 2)
                for row in self.pre._calib_rows:
                    row_scaled = self.pre.transform(row)
                    self.scorer.observe(row_scaled, calibration=True)
                self.scorer.finalize_calibration()

                # Use calibrated ensemble score distribution for thresholds.
                self.calib_scores = list(self.scorer._calib_ensemble)
                self._update_threshold_objects()

                self.calibrating = False

            scores = {"graph": 0.0, "lstm": 0.0, "if": 0.0, "ensemble": 0.0}
            thr = float("nan")
            in_alert = False
            closed_event = None
            contributions = {s: 0.0 for s in SENSOR_NAMES}
            top = []
        else:
            x_scaled = self.pre.transform(x_raw)
            scores = self.scorer.observe(x_scaled, calibration=False)

            thr = self.compute_threshold(threshold_method, target_fpr)
            contributions = self.scorer.explain(x_scaled)
            top = [s for s, _c in sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:5]]

            in_alert, closed_event = self.alert_mgr.update(
                t=t, score=scores["ensemble"], thr=thr, scenario=scenario, top_sensors=top
            )

        self.step_idx += 1

        # persist history for plots/drilldown
        self.history_t.append(t)
        self.history_values.append(x_raw)
        self.history_scenario.append(scenario)
        self.history_score.append(scores["ensemble"])
        self.history_thr.append(thr if not math.isnan(thr) else 0.0)

        return {
            "t": t,
            "raw": raw,
            "values": modified,
            "scenario": scenario,
            "is_anomaly": is_anom,
            "scores": scores,
            "thr": thr,
            "in_alert": in_alert,
            "closed_event": closed_event,
            "contrib": contributions,
            "top_sensors": top,
        }
# -----------------------------
# 8) Bokeh UI (Operator-style)
# -----------------------------

cfg = DemoConfig()
demo = CIADDemo(cfg)

# Data sources
score_source = ColumnDataSource(data=dict(t=[], ensemble=[], thr=[], graph=[], lstm=[], iforest=[], alert=[]))
contrib_source = ColumnDataSource(data=dict(sensor=SENSOR_NAMES, contrib=[0.0] * S))
alerts_source = ColumnDataSource(data=dict(
    id=[],
    start=[],
    end=[],
    duration=[],
    peak=[],
    scenario=[],
    top_sensors=[],
))

drill_source = ColumnDataSource(data=dict(xs=[], ys=[], sensor=[]))

# Controls
toggle_run = Toggle(label="RUN / PAUSE", button_type="success", active=False)
btn_reset = Button(label="RESET", button_type="warning")
select_site = Select(title="Site (OOD setting)", value="Site A", options=["Site A", "Site B"])
select_threshold = Select(title="Thresholding", value=cfg.threshold_method, options=["EVT-POT", "Quantile"])
slider_fpr = Slider(title="Target false positive rate (per tick)", start=0.001, end=0.05, value=cfg.target_fpr, step=0.001)
select_alert = Select(title="Alert drill-down", value="(none)", options=["(none)"])

# Status panels
status_div = Div(text="", sizing_mode="stretch_width")
arch_div = Div(text="", sizing_mode="stretch_width")
alert_detail_div = Div(text="", sizing_mode="stretch_width")

def fmt_float(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{nd}f}"

def build_status(t: float, scenario: str, scores: Dict[str, float], thr: float, in_alert: bool, calibrating: bool) -> str:
    method = select_threshold.value
    fpr = slider_fpr.value

    badge = "<span style='padding:3px 8px;border-radius:10px;background:#e74c3c;color:white;font-weight:600;'>ALERT</span>" if in_alert else \
            "<span style='padding:3px 8px;border-radius:10px;background:#2ecc71;color:white;font-weight:600;'>OK</span>"

    calib_badge = "<span style='padding:3px 8px;border-radius:10px;background:#34495e;color:white;font-weight:600;'>CALIBRATING</span>" if calibrating else ""

    return f"""
    <div style="font-family:ui-sans-serif,system-ui; line-height:1.35;">
      <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
        <div style="font-size:18px; font-weight:700;">CIAD Live Monitor</div>
        {badge} {calib_badge}
        <div style="margin-left:auto; font-size:12px; color:#555;">SCADA stream → preprocess → ensemble detector → thresholds → alerts</div>
      </div>
      <div style="margin-top:8px; display:grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap:10px;">
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Site</div><div style="font-weight:700">{select_site.value}</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Active scenario</div><div style="font-weight:700">{scenario}</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Ensemble score</div><div style="font-weight:700">{fmt_float(scores.get("ensemble",0.0),3)}</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Threshold</div><div style="font-weight:700">{fmt_float(thr,3)}</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Threshold mode</div><div style="font-weight:700">{method}</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Target FPR</div><div style="font-weight:700">{fpr:.3f}</div></div>
      </div>
    </div>
    """

def build_architecture_text() -> str:
    # A compact "what this system is" panel for demo narration.
    sensor_lines = "".join([f"<li><b>{n}</b> ({SENSOR_UNITS[n]})</li>" for n in SENSOR_NAMES])
    edges_lines = "".join([f"<li>{a} ↔ {b}</li>" for a,b in GRAPH_EDGES[:10]]) + ("<li>…</li>" if len(GRAPH_EDGES) > 10 else "")
    return f"""
    <div style="font-family:ui-sans-serif,system-ui; line-height:1.35;">
      <h3 style="margin:0 0 6px 0;">CIAD demo scope</h3>
      <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap:14px;">
        <div style="padding:12px;border:1px solid #ddd;border-radius:10px;">
          <div style="font-weight:700;">Threat model</div>
          <ul style="margin:6px 0 0 18px;">
            <li>Equipment degradation: drift, intermittent noise, stuck-at</li>
            <li>Cyber manipulation: replay, stealthy ramp, coordinated multi-sensor</li>
          </ul>
          <div style="font-weight:700;margin-top:10px;">Pipeline</div>
          <ol style="margin:6px 0 0 18px;">
            <li>Streaming ingest (simulated OPC-UA/Modbus → gateway → historian)</li>
            <li>Preprocess: missingness → impute → scale → window</li>
            <li>Models: Graph-AE stub + LSTM-AE stub + Isolation Forest</li>
            <li>Thresholding: Quantile + EVT-POT (operator-tunable FPR)</li>
            <li>Alert eventization + explanation (top sensors) + audit log stub</li>
          </ol>
        </div>
        <div style="padding:12px;border:1px solid #ddd;border-radius:10px;">
          <div style="font-weight:700;">Schema (sensors)</div>
          <ul style="margin:6px 0 0 18px; columns:2;">
            {sensor_lines}
          </ul>
          <div style="font-weight:700;margin-top:10px;">Topology (subset)</div>
          <ul style="margin:6px 0 0 18px;">
            {edges_lines}
          </ul>
        </div>
      </div>
    </div>
    """

arch_div.text = build_architecture_text()

# Plots
score_plot = figure(
    title="Anomaly score (ensemble) with threshold",
    height=300,
    sizing_mode="stretch_width",
    x_axis_label="t (sec)",
    y_axis_label="score",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
score_plot.line("t", "ensemble", source=score_source, line_width=2, legend_label="ensemble")
score_plot.line("t", "thr", source=score_source, line_width=2, line_dash="dashed", legend_label="threshold")
score_plot.line("t", "graph", source=score_source, line_width=1, line_alpha=0.6, legend_label="graph-AE stub")
score_plot.line("t", "lstm", source=score_source, line_width=1, line_alpha=0.6, legend_label="lstm-AE stub")
score_plot.line("t", "iforest", source=score_source, line_width=1, line_alpha=0.6, legend_label="isolation forest")
score_plot.legend.location = "top_left"
score_plot.legend.click_policy = "hide"

contrib_plot = figure(
    title="Per-alert sensor contribution (explainability)",
    height=300,
    sizing_mode="stretch_width",
    x_range=SENSOR_NAMES,
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
contrib_plot.vbar(x="sensor", top="contrib", width=0.85, source=contrib_source)
contrib_plot.xaxis.major_label_orientation = 1.0

# Key sensors (scaled) plot
key_tags = ["F_OUTLET_FLOW", "P1_PRESSURE", "TANK_LEVEL", "TURBIDITY"]
key_plot_source = ColumnDataSource(data=dict(t=[], **{k: [] for k in key_tags}))
key_plot = figure(
    title="Key sensor telemetry (scaled)",
    height=300,
    sizing_mode="stretch_width",
    x_axis_label="t (sec)",
    y_axis_label="scaled value",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
for i, tag in enumerate(key_tags):
    key_plot.line("t", tag, source=key_plot_source, line_width=2, legend_label=tag, color=Category10[10][i])
key_plot.legend.location = "top_left"
key_plot.legend.click_policy = "hide"

# Drill-down plot (multi-line around an alert)
drill_plot = figure(
    title="Alert drill-down (top sensors around event window)",
    height=320,
    sizing_mode="stretch_width",
    x_axis_label="t (sec)",
    y_axis_label="raw value",
    tools="pan,wheel_zoom,box_zoom,reset,save",
)
drill_plot.multi_line(xs="xs", ys="ys", legend_field="sensor", source=drill_source, line_width=2)
drill_plot.legend.location = "top_left"
drill_plot.legend.click_policy = "hide"

# Alerts table
columns = [
    TableColumn(field="id", title="ID", formatter=NumberFormatter(format="0")),
    TableColumn(field="start", title="Start (s)", formatter=NumberFormatter(format="0.0")),
    TableColumn(field="end", title="End (s)", formatter=NumberFormatter(format="0.0")),
    TableColumn(field="duration", title="Dur (s)", formatter=NumberFormatter(format="0.0")),
    TableColumn(field="peak", title="Peak", formatter=NumberFormatter(format="0.000")),
    TableColumn(field="scenario", title="Scenario", formatter=StringFormatter()),
    TableColumn(field="top_sensors", title="Top sensors", formatter=StringFormatter()),
]
alerts_table = DataTable(source=alerts_source, columns=columns, height=240, sizing_mode="stretch_width", selectable=True, index_position=None)


def build_alert_detail(evt: Optional[AlertEvent]) -> str:
    if evt is None:
        return "<div style='font-family:ui-sans-serif,system-ui;color:#555;'>Select an alert row for details.</div>"
    tops = ", ".join(evt.top_sensors)
    return f"""
    <div style="font-family:ui-sans-serif,system-ui; line-height:1.35;">
      <div style="font-weight:800; font-size:16px;">Alert #{evt.id}</div>
      <div style="margin-top:6px; display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:10px;">
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Start</div><div style="font-weight:700">{evt.start_t:.1f}s</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">End</div><div style="font-weight:700">{evt.end_t:.1f}s</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Duration</div><div style="font-weight:700">{evt.end_t-evt.start_t:.1f}s</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Peak score</div><div style="font-weight:700">{evt.peak_score:.3f}</div></div>
        <div style="padding:10px;border:1px solid #ddd;border-radius:10px;"><div style="color:#666;font-size:12px;">Scenario</div><div style="font-weight:700">{evt.scenario}</div></div>
      </div>
      <div style="margin-top:8px; padding:10px;border:1px solid #ddd;border-radius:10px;">
        <div style="color:#666;font-size:12px;">Top contributing sensors</div>
        <div style="font-weight:700">{tops}</div>
      </div>
    </div>
    """

alert_detail_div.text = build_alert_detail(None)

# -------------
# UI callbacks
# -------------

def _rebuild_alert_select():
    opts = ["(none)"] + [f"#{e.id} [{e.start_t:.0f}-{e.end_t:.0f}s]" for e in demo.alert_mgr.events]
    select_alert.options = opts
    if select_alert.value not in opts:
        select_alert.value = "(none)"


def _update_drilldown_for_event(evt: AlertEvent):
    # Window around event
    pad = 10.0
    t0 = max(0.0, evt.start_t - pad)
    t1 = evt.end_t + pad

    # find indices
    ts = np.asarray(demo.history_t, dtype=np.float64)
    mask = (ts >= t0) & (ts <= t1)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        drill_source.data = dict(xs=[], ys=[], sensor=[])
        return

    # plot raw values for top sensors
    xs = []
    ys = []
    names = []
    for tag in evt.top_sensors:
        i = NAME_TO_IDX[tag]
        series = [float(demo.history_values[k][i]) for k in idx]
        xs.append([float(ts[k]) for k in idx])
        ys.append(series)
        names.append(tag)

    drill_source.data = dict(xs=xs, ys=ys, sensor=names)
    drill_plot.title.text = f"Alert drill-down (top sensors around event #{evt.id})"


def on_alert_select_change(attr, old, new):
    if select_alert.value == "(none)":
        alert_detail_div.text = build_alert_detail(None)
        drill_source.data = dict(xs=[], ys=[], sensor=[])
        drill_plot.title.text = "Alert drill-down (top sensors around event window)"
        return

    # parse #id from value
    try:
        s = select_alert.value
        eid = int(s.split()[0].replace("#", ""))
    except Exception:
        return

    evt = next((e for e in demo.alert_mgr.events if e.id == eid), None)
    alert_detail_div.text = build_alert_detail(evt)
    if evt is not None:
        _update_drilldown_for_event(evt)


select_alert.on_change("value", on_alert_select_change)


def on_table_selection(attr, old, new):
    inds = alerts_source.selected.indices
    if not inds:
        return
    i = inds[0]
    try:
        eid = int(alerts_source.data["id"][i])
    except Exception:
        return
    # set select box to match
    for opt in select_alert.options:
        if opt.startswith(f"#{eid} "):
            select_alert.value = opt
            return


alerts_source.selected.on_change("indices", on_table_selection)


def on_site_change(attr, old, new):
    demo.set_site(select_site.value)
    score_source.data = dict(t=[], ensemble=[], thr=[], graph=[], lstm=[], iforest=[], alert=[])
    contrib_source.data = dict(sensor=SENSOR_NAMES, contrib=[0.0] * S)
    alerts_source.data = dict(id=[], start=[], end=[], duration=[], peak=[], scenario=[], top_sensors=[])
    key_plot_source.data = dict(t=[], **{k: [] for k in key_tags})
    drill_source.data = dict(xs=[], ys=[], sensor=[])
    alert_detail_div.text = build_alert_detail(None)
    select_alert.options = ["(none)"]
    select_alert.value = "(none)"


select_site.on_change("value", on_site_change)


def on_reset():
    demo.reset()
    score_source.data = dict(t=[], ensemble=[], thr=[], graph=[], lstm=[], iforest=[], alert=[])
    contrib_source.data = dict(sensor=SENSOR_NAMES, contrib=[0.0] * S)
    alerts_source.data = dict(id=[], start=[], end=[], duration=[], peak=[], scenario=[], top_sensors=[])
    key_plot_source.data = dict(t=[], **{k: [] for k in key_tags})
    drill_source.data = dict(xs=[], ys=[], sensor=[])
    alert_detail_div.text = build_alert_detail(None)
    select_alert.options = ["(none)"]
    select_alert.value = "(none)"


btn_reset.on_click(on_reset)


# Main streaming loop (Bokeh periodic callback)
def tick():
    if not toggle_run.active:
        # still update status on pause
        if demo.history_t:
            t = demo.history_t[-1]
            scores = {"ensemble": demo.history_score[-1], "graph": 0, "lstm": 0, "if": 0}
            thr = demo.history_thr[-1]
            status_div.text = build_status(t, demo.history_scenario[-1], scores, thr, False, demo.calibrating)
        else:
            status_div.text = build_status(0.0, "—", {"ensemble": 0.0}, float("nan"), False, demo.calibrating)
        return

    out = demo.step(select_threshold.value, slider_fpr.value)
    t = float(out["t"])
    scenario = str(out["scenario"])
    scores = out["scores"]
    thr = float(out["thr"]) if not demo.calibrating else float("nan")
    in_alert = bool(out["in_alert"]) if not demo.calibrating else False

    # status
    status_div.text = build_status(t, scenario, scores, thr, in_alert, demo.calibrating)

    # score plot update
    score_source.stream({
        "t": [t],
        "ensemble": [scores["ensemble"]],
        "thr": [thr if not math.isnan(thr) else np.nan],
        "graph": [scores["graph"]],
        "lstm": [scores["lstm"]],
        "iforest": [scores["if"]],
        "alert": [1 if in_alert else 0],
    }, rollover=500)

    # contributions plot update (only after calibration)
    if not demo.calibrating:
        c = out["contrib"]
        contrib_source.data = dict(sensor=list(c.keys()), contrib=list(c.values()))

    # key sensor plot (scaled)
    if demo.pre.is_fitted:
        x_scaled = demo.pre.transform(np.array([out["values"][k] for k in SENSOR_NAMES], dtype=np.float32))
        key_plot_source.stream({"t": [t], **{k: [float(x_scaled[NAME_TO_IDX[k]])] for k in key_tags}}, rollover=500)

    # new alert events
    closed: Optional[AlertEvent] = out.get("closed_event", None)
    if closed is not None:
        alerts_source.stream({
            "id": [closed.id],
            "start": [closed.start_t],
            "end": [closed.end_t],
            "duration": [closed.end_t - closed.start_t],
            "peak": [closed.peak_score],
            "scenario": [closed.scenario],
            "top_sensors": [", ".join(closed.top_sensors)],
        }, rollover=100)

        _rebuild_alert_select()

    # demo stop after duration
    if t >= cfg.demo_duration_sec:
        # Force-close an active event so it shows in the feed.
        forced = demo.alert_mgr.force_close(t)
        if forced is not None:
            alerts_source.stream({
                "id": [forced.id],
                "start": [forced.start_t],
                "end": [forced.end_t],
                "duration": [forced.end_t - forced.start_t],
                "peak": [forced.peak_score],
                "scenario": [forced.scenario],
                "top_sensors": [", ".join(forced.top_sensors)],
            }, rollover=100)
            _rebuild_alert_select()

        toggle_run.active = False

        m = event_metrics(demo.alert_mgr.events, demo.truth)
        # Append to status with benchmark summary.
        status_div.text = status_div.text + f"""
        <div style="margin-top:10px; font-family:ui-sans-serif,system-ui;">
          <div style="padding:12px;border:1px solid #ddd;border-radius:10px;">
            <div style="font-weight:800;">2-minute benchmark summary</div>
            <div style="margin-top:6px; display:grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap:10px;">
              <div><div style="color:#666;font-size:12px;">TP</div><div style="font-weight:700">{m["TP"]}</div></div>
              <div><div style="color:#666;font-size:12px;">FP</div><div style="font-weight:700">{m["FP"]}</div></div>
              <div><div style="color:#666;font-size:12px;">FN</div><div style="font-weight:700">{m["FN"]}</div></div>
              <div><div style="color:#666;font-size:12px;">Precision</div><div style="font-weight:700">{m["precision"]:.2f}</div></div>
              <div><div style="color:#666;font-size:12px;">Recall</div><div style="font-weight:700">{m["recall"]:.2f}</div></div>
              <div><div style="color:#666;font-size:12px;">F1</div><div style="font-weight:700">{m["f1"]:.2f}</div></div>
            </div>
          </div>
        </div>
        """


# Layout
controls = column(toggle_run, btn_reset, select_site, select_threshold, slider_fpr, select_alert, sizing_mode="stretch_width")

live_left = column(status_div, controls, sizing_mode="stretch_width")
live_right = column(score_plot, row(contrib_plot, key_plot, sizing_mode="stretch_width"), alerts_table, alert_detail_div, drill_plot, sizing_mode="stretch_width")

live_panel = Panel(child=row(live_left, live_right, sizing_mode="stretch_width"), title="Live monitor")
arch_panel = Panel(child=column(arch_div, sizing_mode="stretch_width"), title="What CIAD is")
tabs = Tabs(tabs=[live_panel, arch_panel])

curdoc().add_root(tabs)
curdoc().title = "CIAD Demo — Critical Infrastructure Anomaly Detector"

# initial status
status_div.text = build_status(0.0, "NORMAL", {"ensemble": 0.0, "graph": 0, "lstm": 0, "if": 0}, float("nan"), False, True)

# periodic callback (2 Hz)
curdoc().add_periodic_callback(tick, int(cfg.dt_sec * 1000))
