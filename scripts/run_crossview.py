"""Cross-view test: train on URFD cam0 (side camera, 70 videos), test on cam1 (ceiling camera, 30 falls).

    python scripts/run_crossview.py --extractor yolo

cam1 is synchronised with cam0 (same frame numbers), so cam0's fall onsets apply. cam1 has no ADLs, so only
fall recall and latency can be measured here; false alarms come from the cam0 LOSO runs (run_view.py).
The alarm rule (thr, k) is picked on cam0 only (inner leave-one-group-out), never on cam1.
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
from fallcv import m1, pose, rule, urfd, windows  # noqa: E402
from fallcv.eval import evaluate  # noqa: E402

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
seqs = urfd.sequences()
falls = [s for s in seqs if s.startswith("fall")]
cam1 = {s: pose.load(f"{args.extractor}_cam1", s) for s in falls}
seen_lying = {}
for s in falls:
    f = lab[(lab.seq == s) & (lab.label == 1)].frame.values - 1
    seen_lying[s] = (~pd.isna(cam1[s][0][f, 11, 0])).mean() if len(f) else float("nan")
print(f"cam1 skeleton found on lying frames: {pd.Series(seen_lying).mean():.0%} (mean over falls)")


def report(name, res):
    d, s = evaluate(res, onsets, lab)
    return dict(method=name, falls_hit=s["falls_hit"], late=s["late_detections"],
                caught_at_all=f"{sum(len(a) > 0 for _, a in res.values())}/{len(res)}",
                pre_onset_alarms=s["pre_onset_false_alarms"], latency_s=s["median_latency_s"])


rows = [report("B0 rule", {s: rule.run(*cam1[s])[:2] for s in falls})]
for sig, n_aug in [("full", 0), ("view_free", 0), ("view_free", args.n_aug)]:
    built = m1.build(args.extractor, 1.0, onsets, sig, n_aug=n_aug)
    X, Y, USE = built[:3]
    AUG = built[3] if n_aug else None
    Xc = {s: windows.window_features(windows.frame_signals(*cam1[s], signals=sig), 1.0) for s in falls}
    for mname, model in MODELS.items():
        rule_ = m1.pick_rule(model, X, Y, USE, groups, seqs, onsets, lab, AUG)
        m = m1.fit_on(model, seqs, X, Y, USE, AUG)
        res = m1.to_results({s: m.predict_proba(Xc[s])[:, 1] for s in falls}, rule_)
        rows.append(report(f"M1 {mname}, {sig}{' + aug' if n_aug else ''}", res))
        print(rows[-1], flush=True)
t = pd.DataFrame(rows)
t.to_csv(ROOT / "results" / f"crossview_{args.extractor}.csv", index=False)
print("\ntrain cam0 (side) -> test cam1 (ceiling), 30 falls:")
print(t.round(2).to_string(index=False))
