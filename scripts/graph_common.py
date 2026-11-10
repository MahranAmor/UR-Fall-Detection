"""Shared by the step 6 graph models: the skeleton graph, clip loading and one scoring function.

Graph: 15 of MediaPipe's 33 joints, 16 bones, one connected component.
  - head = nose only: the 10 other face points are reliable but say nothing more about a fall;
    nose-to-both-shoulders edges connect the head to the body (it is a separate part in MediaPipe's graph)
  - fingers dropped: 20-30% filled or visibility < 0.5 in clips, and they add little to the wrists
  - one point per foot (foot index): heel and foot index sit next to the ankle
Legs are kept although ~30% of their joint-frames are unreliable: they are central to what a fall looks like.

Scoring is the same as scripts/baseline_depth_clips.py, so all models compare fairly: splits/splits.json
(3 seeds x 5 folds, clips follow their sequence), threshold 0.5, clip-level recall / precision / F1 / AUC,
and sequence-level scores where a sequence is a fall if any of its clips is.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / "data" / "processed" / "clips.npz"
RESULTS = REPO / "results"

# MediaPipe index -> name, in node order
JOINTS = {0: "nose", 11: "l_shoulder", 12: "r_shoulder", 13: "l_elbow", 14: "r_elbow", 15: "l_wrist",
          16: "r_wrist", 23: "l_hip", 24: "r_hip", 25: "l_knee", 26: "r_knee", 27: "l_ankle", 28: "r_ankle",
          31: "l_foot", 32: "r_foot"}
MP_INDEX = list(JOINTS)
NAMES = list(JOINTS.values())
N = {n: i for i, n in enumerate(NAMES)}
EDGES = [(N[a], N[b]) for a, b in [
    ("nose", "l_shoulder"), ("nose", "r_shoulder"), ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_foot"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_foot")]]
# node index of each joint's mirror image (for left-right flip augmentation)
MIRROR = [N[n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")] for n in NAMES]


def adjacency():
    A = np.zeros((len(NAMES), len(NAMES)), np.float32)
    for a, b in EDGES:
        A[a, b] = A[b, a] = 1
    return A


def load_clips():
    c = np.load(CLIPS)
    return dict(pose=c["pose"][:, :, MP_INDEX], traj=c["traj"], visibility=c["visibility"][:, :, MP_INDEX],
                label=c["label"].astype(int), seq=c["seq"], start=c["start"])


def evaluate(name, fit_predict, data, verbose=True):
    """fit_predict(train_idx, test_idx, seed) -> fall probability for each test clip."""
    splits = json.loads((REPO / "splits" / "splits.json").read_text(encoding="utf-8"))
    y, seq = data["label"], pd.Series(data["seq"])
    scores, preds = [], []
    for sp in splits["splits"]:
        for f in sp["folds"]:
            tr = np.where(seq.isin(f["train"]))[0]
            te = np.where(seq.isin(f["test"]))[0]
            p = np.asarray(fit_predict(tr, te, sp["seed"] * 10 + f["fold"]), float)
            yp = (p >= 0.5).astype(int)
            s = pd.DataFrame({"seq": seq[te].values, "p": p}).groupby("seq").p.max()
            st, sp_ = s.index.str.startswith("fall").astype(int), (s.values >= 0.5).astype(int)
            scores.append({"model": name, "seed": sp["seed"], "fold": f["fold"],
                           "clip_recall": recall_score(y[te], yp), "clip_precision": precision_score(y[te], yp, zero_division=0),
                           "clip_f1": f1_score(y[te], yp), "clip_auc": roc_auc_score(y[te], p),
                           "seq_recall": recall_score(st, sp_), "seq_precision": precision_score(st, sp_, zero_division=0),
                           "seq_f1": f1_score(st, sp_)})
            preds.append(pd.DataFrame({"model": name, "seed": sp["seed"], "fold": f["fold"], "seq": seq[te].values,
                                       "start": data["start"][te], "y": y[te], "score": p}))
            if verbose:
                print(f"  seed {sp['seed']} fold {f['fold']}: clip F1 {scores[-1]['clip_f1']:.3f}, AUC {scores[-1]['clip_auc']:.3f}", flush=True)
    s = pd.DataFrame(scores)
    metrics = [c for c in s.columns if c.startswith(("clip_", "seq_"))]
    summary = s.groupby("model")[metrics].agg(["mean", "std"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / f"{name}_summary.csv", lineterminator="\n")
    pd.concat(preds).to_csv(RESULTS / f"{name}_predictions.csv", index=False, lineterminator="\n", float_format="%.4f")
    print(f"\n{name}: mean (std) over 3 seeds x 5 folds")
    print(pd.DataFrame({m: [f"{summary[m + '_mean'].iloc[0]:.3f} ({summary[m + '_std'].iloc[0]:.3f})"] for m in metrics},
                       index=[name]).T.to_string(header=False))
    return summary
