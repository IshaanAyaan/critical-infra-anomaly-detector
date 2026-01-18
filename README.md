# Critical Infrastructure Anomaly Detector (CIAD)

A reproducible **end-to-end anomaly detection pipeline** for critical infrastructure telemetry (SCADA / sensor logs):
- robust preprocessing + windowing
- multiple detectors (deep + classical)
- operational thresholding (Quantile or POT/EVT)
- event-level evaluation + top-sensor explanations
- batch inference + optional streaming API

This repo is **defensive**: it detects abnormal behavior and outages/attacks early.

---

## Quick start (runs without external datasets)

```bash
pip install -r requirements.txt
python examples/make_synthetic_scada.py --out data/synth.csv
python -m infra_anomdet.cli train --csv data/synth.csv --out artifacts/synth --model stgtae --epochs 5
python -m infra_anomdet.cli infer --csv data/synth.csv --artifacts artifacts/synth --out reports/synth_scored.csv
```

---

## Using a real SCADA dataset (BATADAL-style)

If your CSV has:
- a timestamp column (e.g. `DATETIME` or `timestamp`)
- many sensor columns (floats/ints)
- an optional label column (e.g. `ATT_FLAG`, where 0=normal, 1=anomaly)

then you can run:

```bash
python -m infra_anomdet.cli train --csv path/to/data.csv --out artifacts/run1 --model stgtae --epochs 20 --label-col ATT_FLAG
python -m infra_anomdet.cli infer --csv path/to/data.csv --artifacts artifacts/run1 --out reports/scored.csv
```

---

## Models included

### Deep unsupervised (time-series)
- **STGTAE**: Spatio-Temporal Graph Transformer Autoencoder (sensor-graph + temporal transformer)
- **TransformerAE**: temporal transformer reconstruction
- **LSTMAE**: seq2seq reconstruction
- **USAD**: dual-decoder reconstruction / reconstruction-of-reconstruction

### Classical baselines
- Isolation Forest (on engineered rolling/window features)
- One-Class SVM (same features)

---

## Thresholding

- `quantile`: simple percentile threshold fit on normal validation scores
- `pot`: Peak-over-Threshold with generalized Pareto tail fit (EVT), targets a false positive rate

---

## Artifacts

Training writes:
- `model.pt` (or `model.pkl` for sklearn)
- `scaler.json`
- `meta.json` (feature names, seq_len, etc.)
- `threshold.json`
- `val_scores.csv`

Inference writes:
- `score`, `threshold`, `is_anomaly`
- `top_sensors` (deep models): sensors with largest reconstruction error at each decision point

---

## Notes

This code is designed to be extendable:
- add multi-modal network-traffic features (pcap → per-time-bin stats) and fuse via late-score fusion
- add drift detection and scheduled retraining
- wrap `infer_stream` for real-time alerts

