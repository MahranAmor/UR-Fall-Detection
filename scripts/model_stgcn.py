"""Step 6.4: GNN, a small spatio-temporal graph convolutional network (ST-GCN).

Usage: python scripts/model_stgcn.py [--no-graph]

Input per joint and frame (7 channels): pose x, y (relative to the frame's hip centre), visibility,
joint velocity x, y, and the clip's hip drop and hip speed (the same value on every joint).
Three blocks, each: spatial graph convolution over the 15-joint skeleton (normalised A + I), then a
temporal convolution over 9 frames, batch norm, ReLU, residual connection, dropout. Channels 32, 32, 64
(stride 2 in time in the last block), then average over joints and frames and one linear layer.

Small data, so: ~50k parameters; fall clips weighted by n_not_fall / n_fall in the loss (keeps 0.5 a
balanced threshold like the other models); augmentation on training clips only (left-right flip with
joints swapped, +-10% scale, small noise); channel normalisation from training clips only; early stopping
on a validation split of the training sequences (20%, never the test fold).

--no-graph replaces the skeleton adjacency with the identity (each joint only sees itself): the same
network without message passing between joints, to test whether the graph helps.
--no-hip drops the two hip channels, so the fall must be recognised from body shape and joint motion.
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import graph_common as g

FPS = 30
EPOCHS = 80
PATIENCE = 15
BATCH = 32
LR = 1e-3
WD = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def features(data, use_hip=True):
    pose, traj, vis = data["pose"], data["traj"], data["visibility"]
    vel = np.gradient(pose, axis=1) * FPS
    parts = [pose, vis[..., None], vel]
    if use_hip:
        hip = np.stack([traj[:, :, 1], np.gradient(traj[:, :, 1], axis=1) * FPS], 2)   # (N, T, 2)
        parts.append(np.repeat(hip[:, :, None, :], pose.shape[2], 2))
    x = np.concatenate(parts, 3)                                                       # (N, T, V, 7 or 5)
    return x.transpose(0, 3, 1, 2).astype(np.float32)                                  # (N, C, T, V)


def norm_adjacency(use_graph):
    A = g.adjacency() if use_graph else np.zeros((15, 15), np.float32)
    A = A + np.eye(len(A), dtype=np.float32)
    d = A.sum(1) ** -0.5
    return torch.tensor(d[:, None] * A * d[None, :])


class Block(nn.Module):
    def __init__(self, cin, cout, A, stride=1, dropout=0.3):
        super().__init__()
        self.register_buffer("A", A)
        self.spatial = nn.Conv2d(cin, cout, 1)
        self.temporal = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(),
                                      nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)), nn.BatchNorm2d(cout))
        self.res = nn.Identity() if cin == cout and stride == 1 else \
            nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
        self.out = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):                                   # x (N, C, T, V)
        y = torch.einsum("nctv,vw->nctw", self.spatial(x), self.A)
        return self.out(self.temporal(y) + self.res(x))


class STGCN(nn.Module):
    def __init__(self, A, cin=7):
        super().__init__()
        self.bn = nn.BatchNorm1d(cin * A.shape[0])
        self.blocks = nn.Sequential(Block(cin, 32, A), Block(32, 32, A), Block(32, 64, A, stride=2))
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        n, c, t, v = x.shape
        x = self.bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t)).reshape(n, c, v, t).permute(0, 1, 3, 2)
        return self.fc(self.blocks(x).mean((2, 3))).squeeze(1)


def augment(x, rng):
    x = x.copy()
    flip = rng.random(len(x)) < 0.5
    x[flip] = x[flip][:, :, :, g.MIRROR]
    x[flip, 0] *= -1                                       # pose x
    x[flip, 3] *= -1                                       # velocity x
    s = rng.uniform(0.9, 1.1, (len(x), 1, 1, 1)).astype(np.float32)
    x[:, [c for c in (0, 1, 3, 4, 5, 6) if c < x.shape[1]]] *= s
    x[:, [0, 1]] += rng.normal(0, 0.02, x[:, [0, 1]].shape).astype(np.float32)
    return x


def train_predict(X, y, seq, tr, te, seed, use_graph):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # validation = 20% of the training sequences, falls and ADLs alike (never the test fold)
    seqs = pd.Series(seq[tr]).unique()
    falls, adls = [s for s in seqs if s.startswith("fall")], [s for s in seqs if s.startswith("adl")]
    val_seqs = set(rng.choice(falls, max(1, len(falls) // 5), replace=False)) | \
        set(rng.choice(adls, max(1, len(adls) // 5), replace=False))
    va = tr[np.isin(seq[tr], list(val_seqs))]
    fit = tr[~np.isin(seq[tr], list(val_seqs))]

    model = STGCN(norm_adjacency(use_graph), cin=X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    pos_weight = torch.tensor((y[fit] == 0).sum() / max(1, (y[fit] == 1).sum()), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    to_t = lambda a: torch.tensor(a, device=DEVICE)
    Xva, yva = to_t(X[va]), y[va]
    best, best_state, wait = -1, None, 0
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(fit)
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(to_t(augment(X[b], rng))), to_t(y[b].astype(np.float32)))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva)).cpu().numpy()
        score = roc_auc_score(yva, pv) if len(np.unique(yva)) == 2 else -float(np.mean((pv - yva) ** 2))
        if score > best + 1e-4:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(to_t(X[te]))).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph", action="store_true", help="identity adjacency: no message passing")
    ap.add_argument("--no-hip", action="store_true", help="drop the hip drop / speed channels")
    args = ap.parse_args()
    data = g.load_clips()
    X, y, seq = features(data, not args.no_hip), data["label"], np.asarray(data["seq"])
    use_graph = not args.no_graph
    n_params = sum(p.numel() for p in STGCN(norm_adjacency(use_graph), cin=X.shape[1]).parameters())
    print(f"device {DEVICE}, {n_params} parameters, graph {'on' if use_graph else 'off (identity)'}, "
          f"hip channels {'off' if args.no_hip else 'on'}")
    name = "stgcn" + ("_no_graph" if args.no_graph else "") + ("_no_hip" if args.no_hip else "")
    g.evaluate(name,
               lambda tr, te, seed: train_predict(X, y, seq, tr, te, seed, use_graph), data, verbose=False)


if __name__ == "__main__":
    main()
