"""Step 6.5: compare every model on the same 3 seeds x 5 folds.

Usage: python scripts/compare_models.py   (after the model scripts)

Recomputes the fold metrics from each results/*_predictions.csv (clip level and sequence level, threshold
0.5, a sequence is a fall if any of its clips is) and compares each model with the depth baseline fold by
fold: the fold spread is about +-0.1, so mean differences alone can mislead. Writes
results/model_comparison.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = {  # file stem, model column value -> display name
    ("baseline_depth_clips", "forest"): "Depth features + random forest (baseline)",
    ("baseline_depth_clips", "logreg"): "Depth features + logistic regression",
    ("wl_kernel", "wl_kernel"): "Graph kernel: WL subtree + SVM",
    ("graph_embedding", "graph_embedding"): "Graph embedding: spectral + bones + random forest",
    ("stgcn", "stgcn"): "GNN: ST-GCN",
    ("stgcn_no_graph", "stgcn_no_graph"): "  ST-GCN without graph (check)",
    ("stgcn_no_hip", "stgcn_no_hip"): "  ST-GCN without hip inputs (check)",
    ("stgcn_no_graph_no_hip", "stgcn_no_graph_no_hip"): "  ST-GCN without graph or hip inputs (check)",
}
REF = "Depth features + random forest (baseline)"


def fold_metrics(p):
    rows = []
    for (seed, fold), d in p.groupby(["seed", "fold"]):
        yp = (d.score >= 0.5).astype(int)
        s = d.groupby("seq").score.max()
        st, sp = s.index.str.startswith("fall").astype(int), (s.values >= 0.5).astype(int)
        rows.append({"seed": seed, "fold": fold, "clip_f1": f1_score(d.y, yp), "clip_auc": roc_auc_score(d.y, d.score),
                     "clip_recall": recall_score(d.y, yp), "seq_recall": recall_score(st, sp), "seq_f1": f1_score(st, sp)})
    return pd.DataFrame(rows).set_index(["seed", "fold"])


def main():
    folds = {}
    for (stem, col), name in MODELS.items():
        f = RESULTS / f"{stem}_predictions.csv"
        if f.exists():
            p = pd.read_csv(f)
            folds[name] = fold_metrics(p[p.model == col])
    ref = folds[REF]
    rows = []
    for name, m in folds.items():
        r = {"model": name}
        for k in ["clip_f1", "clip_auc", "clip_recall", "seq_recall", "seq_f1"]:
            r[k] = f"{m[k].mean():.3f} ({m[k].std():.3f})"
        if name != REF:
            d = (m.clip_f1 - ref.clip_f1)
            r["clip_f1_vs_baseline"] = f"{d.mean():+.3f}, better in {(d > 0).sum()}/{len(d)} folds"
        rows.append(r)
    t = pd.DataFrame(rows).set_index("model")
    t.to_csv(RESULTS / "model_comparison.csv", lineterminator="\n")
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(t.to_string())


if __name__ == "__main__":
    main()
