"""Step 5.6: the depth-feature baseline scored on the same clips as the skeleton models.

Usage: python scripts/baseline_depth_clips.py   (after scripts/make_clips.py)

For each kept clip in results/clips_index.csv, the 8 depth features of the same 45 frames are summarised
(mean, std, min, max, start-to-end change, largest per-frame change, after 5-frame smoothing) and scored
with logistic regression and a random forest on splits/splits.json (clips follow their sequence's fold).

Sequence-level the whole-sequence baseline reached F1 0.985 largely because every fall video ends on the
floor; per clip, the model has to recognise the fall itself, which is the comparison step 6 needs.
Reports clip-level and sequence-level scores (a sequence is a fall if any of its clips is).
Writes results/baseline_depth_clips_summary.csv and results/baseline_depth_clips_predictions.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import baseline_depth as bd

REPO = Path(__file__).resolve().parent.parent
WINDOW = 45


def clip_features(frames, index):
    by_seq = {s: d.set_index("frame") for s, d in frames.groupby("seq")}
    rows = []
    for _, r in index.iterrows():
        d = by_seq[r.seq].reindex(range(r.start, r.start + WINDOW))
        x = d[bd.FEATS].rolling(bd.SMOOTH, center=True, min_periods=1).mean()
        step = x.diff().abs()
        f = {}
        for c in bd.FEATS:
            f[f"{c}_mean"], f[f"{c}_std"] = x[c].mean(), x[c].std()
            f[f"{c}_min"], f[f"{c}_max"] = x[c].min(), x[c].max()
            f[f"{c}_delta"] = x[c].iloc[-1] - x[c].iloc[0]
            f[f"{c}_maxstep"] = step[c].max()
        rows.append(f)
    return pd.DataFrame(rows)


def main():
    index = pd.read_csv(REPO / "results" / "clips_index.csv")
    index = index[index.kept].reset_index(drop=True)
    splits = json.loads((REPO / "splits" / "splits.json").read_text(encoding="utf-8"))
    X = clip_features(bd.load_frames(), index)
    assert not X.isna().any().any()
    y = index.label.values
    scores, preds = [], []
    for name, fn in bd.models().items():
        for sp in splits["splits"]:
            for f in sp["folds"]:
                tr, te = index.seq.isin(f["train"]).values, index.seq.isin(f["test"]).values
                m = fn().fit(X[tr], y[tr])
                p = m.predict_proba(X[te])[:, 1]
                yp = (p >= 0.5).astype(int)
                seq = pd.DataFrame({"seq": index.seq[te], "y": y[te], "p": p}).groupby("seq").agg(y=("y", "max"), p=("p", "max"))
                seq_true = seq.index.str.startswith("fall").astype(int)
                seq_pred = (seq.p >= 0.5).astype(int)
                scores.append({"model": name, "seed": sp["seed"], "fold": f["fold"],
                               "clip_recall": recall_score(y[te], yp), "clip_precision": precision_score(y[te], yp, zero_division=0),
                               "clip_f1": f1_score(y[te], yp), "clip_auc": roc_auc_score(y[te], p),
                               "seq_recall": recall_score(seq_true, seq_pred),
                               "seq_precision": precision_score(seq_true, seq_pred, zero_division=0),
                               "seq_f1": f1_score(seq_true, seq_pred)})
                preds.append(pd.DataFrame({"model": name, "seed": sp["seed"], "fold": f["fold"], "seq": index.seq[te],
                                           "start": index.start[te], "y": y[te], "score": p}))
    s = pd.DataFrame(scores)
    metrics = [c for c in s.columns if c.startswith(("clip_", "seq_"))]
    summary = s.groupby("model")[metrics].agg(["mean", "std"]).round(3)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(REPO / "results" / "baseline_depth_clips_summary.csv", lineterminator="\n")
    pd.concat(preds).to_csv(REPO / "results" / "baseline_depth_clips_predictions.csv", index=False,
                            lineterminator="\n", float_format="%.4f")
    pd.set_option("display.width", 200)
    print(f"{len(index)} clips ({int(y.sum())} fall); mean (std) over 3 seeds x 5 folds, threshold 0.5\n")
    print(pd.DataFrame({m: summary[f"{m}_mean"].map("{:.3f}".format) + " (" + summary[f"{m}_std"].map("{:.3f}".format) + ")"
                        for m in metrics}).T.to_string())


if __name__ == "__main__":
    main()
