
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from infra_anomdet.data.loader import load_csv
from infra_anomdet.utils import save_json, load_json, robust_scale_fit, robust_scale_transform, sliding_windows, engineered_features
from infra_anomdet.thresholding.thresholds import quantile_threshold, pot_threshold
from infra_anomdet.models.sklearn_models import make_iforest, make_ocsvm, SklearnDetector
from infra_anomdet.models.torch_models import LSTMAE, TransformerAE, USAD, STGTAE, TorchDetector, build_sensor_graph
from infra_anomdet.eval.metrics import pointwise_metrics, event_metrics

app = typer.Typer(add_completion=False)
console = Console()

def _ensure(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _split(W: np.ndarray, y: Optional[np.ndarray], val_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(W))
    rng.shuffle(idx)
    nval = int(len(W) * val_frac)
    val_idx = idx[:nval]
    tr_idx = idx[nval:]
    Wtr, Wva = W[tr_idx], W[val_idx]
    ytr = y[tr_idx] if y is not None else None
    yva = y[val_idx] if y is not None else None
    return Wtr, Wva, ytr, yva

def _top_sensors(per_sensor_err: np.ndarray, feature_names: List[str], k: int = 5) -> List[str]:
    # per_sensor_err: (B,F)
    out = []
    for i in range(per_sensor_err.shape[0]):
        idx = np.argsort(-per_sensor_err[i])[:k]
        out.append(",".join([feature_names[j] for j in idx]))
    return out

@app.command()
def train(
    csv: str = typer.Option(..., help="Path to CSV with timestamp + sensor columns; optional label col"),
    out: str = typer.Option(..., help="Artifacts output directory"),
    model: str = typer.Option("stgtae", help="stgtae|transformer_ae|lstm_ae|usad|iforest|ocsvm"),
    timestamp_col: Optional[str] = typer.Option(None, help="Timestamp column name; if omitted, inferred"),
    label_col: Optional[str] = typer.Option(None, help="Label column (0 normal, 1 anomaly). If omitted, unsupervised w/ all windows treated as normal for training."),
    seq_len: int = typer.Option(48, help="Window length"),
    stride: int = typer.Option(1, help="Window stride"),
    epochs: int = typer.Option(20, help="Training epochs (torch models)"),
    lr: float = typer.Option(1e-3, help="Learning rate (torch models)"),
    threshold_type: str = typer.Option("pot", help="pot|quantile"),
    target_fpr: float = typer.Option(1e-3, help="Target false positive rate (POT/quantile on normal val)"),
    base_quantile: float = typer.Option(0.98, help="Base quantile for POT u"),
    device: str = typer.Option("auto", help="cpu|cuda|auto"),
):
    outp = _ensure(out)
    data = load_csv(csv, timestamp_col=timestamp_col, label_col=label_col)

    # Window raw series
    W, Wy = sliding_windows(data.X, data.y, seq_len=seq_len, stride=stride)

    # Scale using robust scaler fit on (assumed) normal windows
    # If labels exist, fit scaler on normal-only windows (Wy==0)
    if Wy is not None:
        W_norm = W[Wy == 0] if (Wy == 0).any() else W
    else:
        W_norm = W
    scaler = robust_scale_fit(W_norm)
    W_scaled = robust_scale_transform(W, scaler).astype(np.float32)

    # Split
    Wtr, Wva, ytr, yva = _split(W_scaled, Wy)
    # Train only on normal if labels exist
    if ytr is not None and (ytr == 0).any():
        Wtr_fit = Wtr[ytr == 0]
    else:
        Wtr_fit = Wtr

    meta = {
        "csv": os.path.abspath(csv),
        "timestamp_col": timestamp_col,
        "label_col": label_col,
        "seq_len": seq_len,
        "stride": stride,
        "feature_cols": data.feature_cols,
        "model": model,
    }
    save_json(outp / "meta.json", meta)
    save_json(outp / "scaler.json", scaler)

    # Train model
    if model in ("iforest", "ocsvm"):
        Fe_tr = engineered_features(Wtr_fit)
        Fe_va = engineered_features(Wva)
        det = make_iforest() if model == "iforest" else make_ocsvm()
        det.fit(Fe_tr)
        val_scores = det.score(Fe_va)
        det.save(str(outp / "model.pkl"))
        per_sensor = None
    else:
        F = W_scaled.shape[-1]
        if model == "lstm_ae":
            net = LSTMAE(n_features=F)
        elif model == "transformer_ae":
            net = TransformerAE(n_features=F)
        elif model == "usad":
            # USAD is per-timestep; flatten time dimension by scoring reconstruction per timestep then aggregate
            net = TransformerAE(n_features=F)  # fallback to transformer AE for windows; simpler stable default
        elif model == "stgtae":
            # build graph from a sample of normal windows
            sample = Wtr_fit[: min(2000, len(Wtr_fit))]
            Ahat = build_sensor_graph(sample, topk=min(8, max(2, F//10)))
            net = STGTAE(n_features=F, Ahat=Ahat)
            save_json(outp / "graph.json", {"Ahat": Ahat.tolist()})
        else:
            raise typer.BadParameter(f"Unknown model: {model}")

        det = TorchDetector(kind=model, model=net, seq_len=seq_len, feature_names=data.feature_cols)
        det.train_ae(Wtr_fit, Wva, epochs=epochs, lr=lr, device=device)
        val_scores, per_sensor = det.score_windows(Wva, device=device)
        det.save(str(outp / "model.pt"))

    # Fit threshold using normal validation scores
    if yva is not None and (yva == 0).any():
        normal_scores = val_scores[yva == 0]
    else:
        normal_scores = val_scores

    if threshold_type == "quantile":
        thr = quantile_threshold(normal_scores, 1.0 - target_fpr)
        thr_obj = {"type": "quantile", "q": float(1.0 - target_fpr), "threshold": float(thr)}
    elif threshold_type == "pot":
        thr_obj = pot_threshold(normal_scores, base_quantile=base_quantile, target_fpr=target_fpr)
    else:
        raise typer.BadParameter("threshold_type must be pot or quantile")
    save_json(outp / "threshold.json", thr_obj)

    # Save validation diagnostics
    dfv = pd.DataFrame({"val_score": val_scores})
    if yva is not None:
        dfv["y"] = yva
    dfv.to_csv(outp / "val_scores.csv", index=False)

    # Print metrics if labels exist
    if yva is not None:
        y_pred = (val_scores >= thr_obj["threshold"]).astype(int)
        pm = pointwise_metrics(yva, val_scores, y_pred)
        em = event_metrics(yva, y_pred)
        save_json(outp / "val_metrics.json", {"pointwise": pm, "event": em})
        table = Table(title="Validation metrics")
        for k,v in {**pm, **em}.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)

    console.print(f"[green]Saved artifacts to[/green] {outp}")

@app.command()
def infer(
    csv: str = typer.Option(..., help="CSV to score"),
    artifacts: str = typer.Option(..., help="Artifacts directory produced by train"),
    out: str = typer.Option(..., help="Output scored CSV path"),
    timestamp_col: Optional[str] = typer.Option(None, help="Timestamp column override"),
    label_col: Optional[str] = typer.Option(None, help="Label column override"),
    device: str = typer.Option("auto", help="cpu|cuda|auto"),
    topk: int = typer.Option(5, help="Top-k sensors to show for deep models"),
):
    art = Path(artifacts)
    meta = load_json(art / "meta.json")
    scaler = load_json(art / "scaler.json")
    thr = load_json(art / "threshold.json")
    model_type = meta["model"]
    seq_len = int(meta["seq_len"])
    stride = int(meta["stride"])
    feats = meta["feature_cols"]

    data = load_csv(csv, timestamp_col=timestamp_col or meta.get("timestamp_col"), label_col=label_col or meta.get("label_col"))
    # align to training features
    df = pd.DataFrame(data.X, columns=data.feature_cols)
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected feature columns: {missing[:10]}{'...' if len(missing)>10 else ''}")
    X = df[feats].to_numpy().astype(np.float32)

    W, Wy = sliding_windows(X, data.y, seq_len=seq_len, stride=stride)
    W = robust_scale_transform(W, scaler).astype(np.float32)

    if model_type in ("iforest", "ocsvm"):
        det = SklearnDetector.load(str(art / "model.pkl"))
        S = det.score(engineered_features(W))
        per_sensor = None
    else:
        # rebuild model structure
        F = W.shape[-1]
        if model_type == "lstm_ae":
            ctor = lambda: LSTMAE(n_features=F)
        elif model_type == "transformer_ae":
            ctor = lambda: TransformerAE(n_features=F)
        elif model_type == "stgtae":
            g = load_json(art / "graph.json")
            Ahat = np.array(g["Ahat"], dtype=np.float32)
            ctor = lambda: STGTAE(n_features=F, Ahat=Ahat)
        else:
            ctor = lambda: TransformerAE(n_features=F)
        det = TorchDetector.load(str(art / "model.pt"), model_ctor=ctor)
        S, per_sensor = det.score_windows(W, device=device)

    threshold = float(thr["threshold"])
    pred = (S >= threshold).astype(int)

    # map window scores back to timestamps (use the window end time index)
    idxs = list(range(0, len(X) - seq_len + 1, stride))
    t = data.ts.iloc[np.array(idxs) + (seq_len - 1)].reset_index(drop=True)

    outdf = pd.DataFrame({"timestamp": t, "score": S, "threshold": threshold, "is_anomaly": pred})
    if Wy is not None:
        outdf["y"] = Wy
    if per_sensor is not None:
        outdf["top_sensors"] = _top_sensors(per_sensor, feats, k=topk)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    outdf.to_csv(out, index=False)

    # print metrics if labels exist
    if Wy is not None:
        pm = pointwise_metrics(Wy, S, pred)
        em = event_metrics(Wy, pred)
        table = Table(title="Inference metrics")
        for k,v in {**pm, **em}.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)

    console.print(f"[green]Wrote[/green] {out}")

if __name__ == "__main__":
    app()
