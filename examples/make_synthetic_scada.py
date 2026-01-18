
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--sensors", type=int, default=12)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    t = np.arange(args.n)

    # base dynamics: correlated sinusoids + random walk drift
    base = np.sin(t/50)[:,None] + 0.3*np.sin(t/9)[:,None]
    corr = rng.normal(size=(args.sensors,args.sensors))
    corr = corr @ corr.T
    corr = corr / np.max(np.abs(corr))
    Z = rng.normal(size=(args.n,args.sensors))
    X = base + Z @ corr * 0.1
    drift = rng.normal(scale=0.003, size=(args.n,args.sensors)).cumsum(axis=0)
    X = X + drift

    # inject visible anomalies: spikes, freezes, and inconsistent coupling
    y = np.zeros(args.n, dtype=int)
    for start in [900, 2100, 3300, 4200]:
        dur = rng.integers(80, 180)
        y[start:start+dur] = 1
        kind = rng.choice(["spike", "freeze", "bias"])
        s = rng.integers(0, args.sensors)
        if kind == "spike":
            X[start:start+dur, s] += rng.normal(3.0, 0.5, size=dur)
        elif kind == "freeze":
            X[start:start+dur, s] = X[start, s]
        else:
            X[start:start+dur, s] += 1.5

    df = pd.DataFrame(X, columns=[f"S{i:02d}" for i in range(args.sensors)])
    df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=args.n, freq="min"))
    df["ATT_FLAG"] = y
    df.to_csv(args.out, index=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
