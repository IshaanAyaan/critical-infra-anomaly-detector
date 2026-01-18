
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def _device(dev: str) -> torch.device:
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

class LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, latent: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers>1 else 0.0)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_dec = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers>1 else 0.0)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        _, (h, _) = self.enc(x)
        z = self.fc_mu(h[-1])
        h0 = torch.tanh(self.fc_dec(z)).unsqueeze(0)
        # repeat hidden state across layers
        h0 = h0.repeat(self.dec.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        # feed a learned constant per step (use h0's last layer repeated across time)
        B,T,_ = x.shape
        inp = h0[-1].unsqueeze(1).repeat(1, T, 1)
        y, _ = self.dec(inp, (h0, c0))
        return self.out(y)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]

class TransformerAE(nn.Module):
    def __init__(self, n_features: int, d_model: int = 128, nhead: int = 8, layers: int = 4, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        h = self.pos(h)
        h = self.enc(h)
        return self.out(h)

class USAD(nn.Module):
    # UnSupervised Anomaly Detection with dual decoders
    def __init__(self, n_features: int, hidden: int = 128, latent: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))
        self.dec2 = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        x1 = self.dec1(z)
        x2 = self.dec2(z)
        x1_2 = self.dec2(self.enc(x1))
        return x1, x2, x1_2

def build_sensor_graph(W: np.ndarray, topk: int = 6) -> np.ndarray:
    # W: (B,T,F) sample of normal windows
    X = W.reshape(-1, W.shape[-1])
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    F = corr.shape[0]
    A = np.zeros((F,F), dtype=np.float32)
    for i in range(F):
        idx = np.argsort(-np.abs(corr[i]))  # descending abs corr
        for j in idx[1:topk+1]:
            A[i,j] = 1.0
            A[j,i] = 1.0
    np.fill_diagonal(A, 1.0)
    # normalize: D^-1/2 A D^-1/2
    d = A.sum(axis=1)
    Dinv = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-6)))
    Ahat = Dinv @ A @ Dinv
    return Ahat.astype(np.float32)

class GraphMix(nn.Module):
    def __init__(self, Ahat: np.ndarray):
        super().__init__()
        self.register_buffer("Ahat", torch.tensor(Ahat))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,F)
        A = self.Ahat
        # mix identity and A
        I = torch.eye(A.size(0), device=A.device)
        M = torch.sigmoid(self.alpha) * A + (1.0 - torch.sigmoid(self.alpha)) * I
        return torch.einsum("ij,btj->bti", M, x)

class STGTAE(nn.Module):
    # Graph mixing + temporal transformer AE
    def __init__(self, n_features: int, Ahat: np.ndarray, d_model: int = 128, nhead: int = 8, layers: int = 3, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.gmix = GraphMix(Ahat)
        self.ae = TransformerAE(n_features=n_features, d_model=d_model, nhead=nhead, layers=layers, dim_ff=dim_ff, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xg = self.gmix(x)
        return self.ae(xg)

@dataclass
class TorchDetector:
    kind: str
    model: nn.Module
    seq_len: int
    feature_names: List[str]

    def train_ae(self, W_train: np.ndarray, W_val: np.ndarray, *, epochs: int = 20, lr: float = 1e-3, batch: int = 128, device: str = "auto") -> Dict[str, float]:
        dev = _device(device)
        self.model.to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(TensorDataset(torch.tensor(W_train)), batch_size=batch, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(W_val)), batch_size=batch, shuffle=False)

        best = float("inf")
        best_state = None
        for ep in range(1, epochs+1):
            self.model.train()
            tr = 0.0
            n = 0
            for (x,) in train_loader:
                x = x.to(dev)
                opt.zero_grad(set_to_none=True)
                y = self.model(x)
                loss = loss_fn(y, x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tr += float(loss.item()) * x.size(0)
                n += x.size(0)
            tr /= max(n, 1)
            self.model.eval()
            vl = 0.0
            vn = 0
            with torch.no_grad():
                for (x,) in val_loader:
                    x = x.to(dev)
                    y = self.model(x)
                    loss = loss_fn(y, x)
                    vl += float(loss.item()) * x.size(0)
                    vn += x.size(0)
            vl /= max(vn, 1)
            if vl < best:
                best = vl
                best_state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_val_mse": float(best)}

    @torch.no_grad()
    def score_windows(self, W: np.ndarray, device: str = "auto") -> Tuple[np.ndarray, np.ndarray]:
        # returns scores (B,), per-sensor errors (B,F)
        dev = _device(device)
        self.model.to(dev)
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(W)), batch_size=256, shuffle=False)
        scores = []
        per_sensor = []
        for (x,) in loader:
            x = x.to(dev)
            y = self.model(x)
            err = (y - x) ** 2  # (B,T,F)
            sens = err.mean(dim=1)  # (B,F)
            s = sens.mean(dim=1)    # (B,)
            scores.append(s.detach().cpu().numpy())
            per_sensor.append(sens.detach().cpu().numpy())
        return np.concatenate(scores), np.concatenate(per_sensor)

    def save(self, path: str) -> None:
        torch.save({
            "kind": self.kind,
            "state_dict": self.model.state_dict(),
            "seq_len": self.seq_len,
            "feature_names": self.feature_names,
        }, path)

    @staticmethod
    def load(path: str, model_ctor) -> "TorchDetector":
        ckpt = torch.load(path, map_location="cpu")
        model = model_ctor()
        model.load_state_dict(ckpt["state_dict"])
        return TorchDetector(kind=ckpt["kind"], model=model, seq_len=ckpt["seq_len"], feature_names=ckpt["feature_names"])
