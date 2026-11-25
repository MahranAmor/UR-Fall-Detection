"""Viewpoint refinements for M1, leave-one-group-out on URFD cam0.

    python scripts/run_view.py --extractor yolo

Rows: full signals (with hip height in the image) vs view-free signals (calibrated, body-relative)
vs view-free + skeleton augmentation (3 randomly transformed cameras per training video).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallcv import m1, pose, urfd  # noqa: E402
from fallcv.eval import evaluate, video_level  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--extractor", default="yolo", choices=pose.EXTRACTORS)
ap.add_argument("--n_aug", type=int, default=3)
args = ap.parse_args()

MODELS = {
    "random forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0)),
    "gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                        l2_regularization=1.0, class_weight="balanced", random_state=0),
}
lab, onsets, groups = urfd.labels(), urfd.fall_onsets(), m1.load_groups()
variants = {
    "full": m1.build(args.extractor, 1.0, onsets, "full") + (None,),
    "view_free": m1.build(args.extractor, 1.0, onsets, "view_free") + (None,),
    "view_free + aug": m1.build(args.extractor, 1.0, onsets, "view_free", n_aug=args.n_aug),
}
rows = []
for vname, (X, Y, USE, AUG) in variants.items():
    for mname, model in MODELS.items():
        res, folds, _ = m1.loso(model, X, Y, USE, groups, onsets, lab, AUG)
        _, s = evaluate(res, onsets, lab)
        rows.append(dict(signals=vname, model=mname, falls_hit=s["falls_hit"], adl_fa=s["adl_false_alarms"],
                         latency_s=s["median_latency_s"], recall=f"{folds.recall.mean():.2f} ± {folds.recall.std():.2f}",
                         **{k: video_level(res)[k] for k in ["videos_right"]}))
        print(rows[-1], flush=True)
t = pd.DataFrame(rows)
t.to_csv(ROOT / "results" / f"view_{args.extractor}.csv", index=False)
print(t.round(2).to_string(index=False))
