"""Step 6.3: graph embedding model, a spectral + edge embedding of the skeleton graph and a random forest.

Usage: python scripts/model_graph_embedding.py

Per frame, the skeleton graph gives:
  spectral   graph Fourier transform of the joint heights: projections onto the eigenvectors of the
             skeleton's Laplacian (low modes = whole-body shape, high modes = local differences between
             connected joints)
  bones      verticality of each of the 16 edges, |cos(angle to vertical)| (1 vertical, 0 flat);
             independent of body size and of facing left or right
  hip path   hip drop since the clip start and its speed
Each per-frame value is summarised over the 45 frames (mean, std, min, max, start-to-end change, largest
per-frame change), the same summaries as the depth baseline. The clip embedding feeds the same random
forest as the depth baseline (500 trees, min_samples_leaf 2, balanced class weights, threshold 0.5).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import graph_common as g

FPS = 30


def laplacian_basis():
    A = g.adjacency()
    L = np.diag(A.sum(1)) - A
    _, U = np.linalg.eigh(L)          # columns ordered from smoothest (constant) to most oscillating
    return U


def summarise(x):
    """x (N, T, F) -> (N, 6F)."""
    step = np.abs(np.diff(x, axis=1)).max(1)
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), x[:, -1] - x[:, 0], step], 1)


def embed(data):
    pose, traj = data["pose"], data["traj"]
    U = laplacian_basis()
    spectral = pose[:, :, :, 1] @ U                                     # (N, T, 15)
    bones = np.stack([pose[:, :, b] - pose[:, :, a] for a, b in g.EDGES], 2)   # (N, T, 16, 2)
    vertical = np.abs(bones[..., 1]) / np.maximum(np.linalg.norm(bones, axis=3), 1e-6)
    hip = np.stack([traj[:, :, 1], np.gradient(traj[:, :, 1], axis=1) * FPS], 2)
    return np.concatenate([summarise(spectral), summarise(vertical), summarise(hip)], 1)


def main():
    data = g.load_clips()
    X = embed(data)
    y = data["label"]
    print(f"embedding: {X.shape[1]} numbers per clip")

    def fit_predict(tr, te, seed):
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                   random_state=seed, n_jobs=-1).fit(X[tr], y[tr])
        return m.predict_proba(X[te])[:, 1]

    g.evaluate("graph_embedding", fit_predict, data, verbose=False)


if __name__ == "__main__":
    main()
