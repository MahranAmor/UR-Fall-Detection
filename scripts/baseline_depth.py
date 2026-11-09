"""Step 2: sequence-level fall / no-fall baseline from the dataset's 8 depth features.

Usage: python scripts/baseline_depth.py

No images are used: the features come precomputed in the label csvs.
Each sequence becomes one row of summary statistics (mean, std, min, max,
first-to-last change, largest per-frame change) of the 8 features.
The `label` column and the sequence length are never model inputs; a
length-only model is scored separately as a control for that shortcut.

Scored on splits/splits.json (5 folds x 3 seeds). Writes:
  results/baseline_depth_summary.csv      mean / std of the metrics over the 15 folds
  results/baseline_depth_predictions.csv  per sequence, fold and seed: score and prediction
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "urfd" / "labels"
OUT = REPO / "results"
FEATS = ["HeightWidthRatio", "MajorMinorRatio", "BBoxOccupancy", "MaxStdXZ", "HHmaxRatio", "H", "D", "P40"]
SMOOTH = 5  # frames, centred moving average


def load_frames():
    lab = pd.concat([pd.read_csv(RAW / f"urfall-cam0-{k}.csv", header=None) for k in ["falls", "adls"]],
                    ignore_index=True)
    lab.columns = ["seq", "frame", "label"] + FEATS
    return lab.drop(columns="label").sort_values(["seq", "frame"]).reset_index(drop=True)


def summarise(frames, smooth):
    rows = {}
    for seq, d in frames.groupby("seq"):
        x = d[FEATS]
        if smooth:
            x = x.rolling(SMOOTH, center=True, min_periods=1).mean()
        # per-frame change; divide by the frame gap so the 3 sequences with missing frames are not distorted
        step = x.diff().div(d.frame.diff(), axis=0).abs()
        r = {}
        for c in FEATS:
            r[f"{c}_mean"], r[f"{c}_std"] = x[c].mean(), x[c].std()
            r[f"{c}_min"], r[f"{c}_max"] = x[c].min(), x[c].max()
            r[f"{c}_delta"] = x[c].iloc[-1] - x[c].iloc[0]
            r[f"{c}_maxstep"] = step[c].max()
        rows[seq] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def models():
    """Factories, so every fold trains a fresh, untouched model."""
    return {
        "logreg": lambda: make_pipeline(StandardScaler(),
                                        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
        "forest": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                                 random_state=0, n_jobs=-1),
    }


def evaluate(X, y, splits, name, model_fn):
    preds, scores = [], []
    for sp in splits["splits"]:
        for f in sp["folds"]:
            m = model_fn()
            m.fit(X.loc[f["train"]], y[f["train"]])
            p = m.predict_proba(X.loc[f["test"]])[:, 1]
            yt, yp = y[f["test"]].values, (p >= 0.5).astype(int)
            scores.append({"model": name, "seed": sp["seed"], "fold": f["fold"],
                           "recall": recall_score(yt, yp), "precision": precision_score(yt, yp, zero_division=0),
                           "f1": f1_score(yt, yp), "accuracy": (yt == yp).mean(), "auc": roc_auc_score(yt, p)})
            preds.append(pd.DataFrame({"model": name, "seed": sp["seed"], "fold": f["fold"],
                                       "seq": f["test"], "y": yt, "score": p, "pred": yp}))
    return pd.DataFrame(scores), pd.concat(preds, ignore_index=True)


def main():
    seqs = pd.read_csv(REPO / "splits" / "sequences.csv").set_index("seq")
    splits = json.loads((REPO / "splits" / "splits.json").read_text(encoding="utf-8"))
    y = seqs.label
    frames = load_frames()

    runs = {}
    for smooth in (False, True):
        X = summarise(frames, smooth).loc[seqs.index]
        assert not X.isna().any().any()
        for mname, fn in models().items():
            runs[f"{mname}{'_smooth' if smooth else ''}"] = (X, fn)
    # control: sequence length only (the shortcut we must not rely on)
    runs["control_length_only"] = (seqs[["n_frames"]].astype(float),
                                   lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")))

    all_scores, all_preds = [], []
    for name, (X, fn) in runs.items():
        s, p = evaluate(X, y, splits, name, fn)
        all_scores.append(s); all_preds.append(p)
    scores = pd.concat(all_scores, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    metrics = ["recall", "precision", "f1", "accuracy", "auc"]
    summary = scores.groupby("model", sort=False)[metrics].agg(["mean", "std"]).round(3)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "baseline_depth_summary.csv", lineterminator="\n")
    preds.to_csv(OUT / "baseline_depth_predictions.csv", index=False, lineterminator="\n", float_format="%.4f")

    pd.set_option("display.width", 200)
    print("Mean (std) over 3 seeds x 5 folds, threshold 0.5, per sequence:\n")
    print(pd.DataFrame({m: summary[f"{m}_mean"].map("{:.3f}".format) + " (" + summary[f"{m}_std"].map("{:.3f}".format) + ")"
                        for m in metrics}).to_string())

    best = summary["f1_mean"].drop("control_length_only").idxmax()
    bp = preds[preds.model == best]
    cm = pd.crosstab(bp.y.map({1: "fall", 0: "adl"}), bp.pred.map({1: "fall", 0: "adl"}),
                     rownames=["true"], colnames=["predicted"]) / len(splits["seeds"])
    print(f"\nBest by F1: {best}. Confusion matrix (sequences, averaged over seeds):\n{cm.round(1).to_string()}")
    wrong = bp[bp.y != bp.pred].groupby(["seq", "y"]).size().rename("times_wrong_of_3").reset_index()
    print(f"\nMisclassified sequences ({best}):\n{wrong.sort_values(['y', 'seq']).to_string(index=False) if len(wrong) else 'none'}")
    if best.startswith("forest"):
        X, _ = runs[best]
        imp = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(X, y).feature_importances_
        print("\nTop features (forest fitted on all 70, for reading only):")
        print(pd.Series(imp, X.columns).nlargest(8).round(3).to_string())


if __name__ == "__main__":
    main()
