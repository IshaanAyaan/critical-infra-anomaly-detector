# CIAD Demo (Critical Infrastructure Anomaly Detector)

## Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

bokeh serve --show ciad_bokeh_demo.py
```

## 2-minute demo timeline (built-in)

- 0–20s: calibration (normal-only)
- 25–35s: SPIKE_QUALITY
- 55–70s: STEALTH_RAMP_FLOW (coordinated multi-sensor)
- 78–86s: STUCK_AT_VALVE1
- 95–105s: REPLAY_OUTLET_VALVE2
- 120s: auto-stop + event-level precision/recall/F1 summary

## What you’ll see

- Live SCADA-like sensor stream (synthetic water plant)
- Injected scenarios: spike, stealthy coordinated ramp, stuck-at, replay
- Ensemble anomaly score + threshold line (Quantile or EVT-POT)
- Explainability: per-alert sensor contribution ranking
- Alert feed + drill-down plot around selected alert

## Notes

This is a demo scaffold; the “Graph Transformer AE” and “LSTM AE” are fast stubs meant to mimic interfaces/behavior for an operator demo.
