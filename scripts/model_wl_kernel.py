"""Step 6.2: graph kernel model, Weisfeiler-Lehman subtree kernel + SVM.

Usage: python scripts/model_wl_kernel.py

Every clip becomes one spatio-temporal graph: every STEP-th frame (15 time steps x 15 joints = 225 nodes),
bone edges inside each time step and a temporal edge from each joint to itself in the next time step.
All skeletons share one topology, so the movement has to live in the node labels. Labels are discrete
and fixed from body proportions (in reference torso lengths), not tuned on the data:
  joint name
  height relative to the hip centre: above (< -0.5), level, below (> 0.5)      lying -> everything level
  hip drop since the clip start: none (< 0.3), partial, full (> 0.8)
  hidden if visibility < 0.3
WL relabels each node from its own label and its neighbours' labels for H rounds; the kernel is the
cosine similarity of the label-count histograms (all rounds). An SVM with a precomputed kernel and
balanced class weights classifies the clips; score = sigmoid(decision value), threshold 0.5 = SVM boundary.
"""
import hashlib

import numpy as np
from sklearn.svm import SVC

import graph_common as g

STEP = 3
H = 3
C = 1.0


def node_labels(pose, traj, vis):
    """pose (T, V, 2), traj (T, 2), vis (T, V) -> list of string labels, node = t * V + v."""
    y = pose[:, :, 1]
    height = np.where(y < -0.5, "up", np.where(y > 0.5, "down", "level"))
    drop = traj[:, 1]
    dropped = np.where(drop < 0.3, "d0", np.where(drop > 0.8, "d2", "d1"))
    out = []
    for t in range(pose.shape[0]):
        for v in range(pose.shape[1]):
            out.append(f"{g.NAMES[v]}|{'hid' if vis[t, v] < 0.3 else height[t, v]}|{dropped[t]}")
    return out


def neighbours(T, V):
    nb = [[] for _ in range(T * V)]
    for t in range(T):
        for a, b in g.EDGES:
            nb[t * V + a].append(t * V + b); nb[t * V + b].append(t * V + a)
        if t + 1 < T:
            for v in range(V):
                nb[t * V + v].append((t + 1) * V + v); nb[(t + 1) * V + v].append(t * V + v)
    return nb


def wl_features(data):
    """Label-count histograms over H WL rounds for every clip (an unsupervised feature map)."""
    pose, traj, vis = data["pose"][:, ::STEP], data["traj"][:, ::STEP], data["visibility"][:, ::STEP]
    T, V = pose.shape[1], pose.shape[2]
    nb = neighbours(T, V)
    vocab, rows = {}, []
    for i in range(len(pose)):
        labels = node_labels(pose[i], traj[i], vis[i])
        counts = {}
        for it in range(H + 1):
            for l in labels:
                k = vocab.setdefault((it, l), len(vocab))
                counts[k] = counts.get(k, 0) + 1
            if it < H:
                labels = [hashlib.md5((labels[n] + "(" + ",".join(sorted(labels[m] for m in nb[n])) + ")")
                                      .encode()).hexdigest()[:12] for n in range(T * V)]
        rows.append(counts)
    X = np.zeros((len(rows), len(vocab)), np.float32)
    for i, c in enumerate(rows):
        X[i, list(c)] = list(c.values())
    return X


def main():
    data = g.load_clips()
    X = wl_features(data)
    X = X[:, (X > 0).sum(0) >= 2]                    # drop patterns seen in a single clip only
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    K = Xn @ Xn.T                                    # normalised WL subtree kernel
    print(f"WL features: {X.shape[1]} patterns (seen in >= 2 clips), {H} rounds, {X.shape[0]} clips")
    y = data["label"]

    def fit_predict(tr, te, seed):
        # Score = sigmoid of the SVM decision value, so 0.5 is the SVM's own class-balanced boundary.
        # Platt scaling (probability=True) recalibrates to the 14% fall rate and undoes the balancing:
        # same AUC, but clip recall 0.62 at 0.5 instead of a balanced threshold like the other models.
        m = SVC(kernel="precomputed", C=C, class_weight="balanced")
        m.fit(K[np.ix_(tr, tr)], y[tr])
        return 1 / (1 + np.exp(-m.decision_function(K[np.ix_(te, tr)])))

    g.evaluate("wl_kernel", fit_predict, data, verbose=False)


if __name__ == "__main__":
    main()
