"""M1: causal sliding-window pose features + gradient boosting, evaluated leave-one-group-out.

    python scripts/run_m1.py --extractor mediapipe
    python scripts/run_m1.py --extractor yolo --win 1.5

See src/fallcv/m1.py for the protocol (alarm rule picked on the 4 training groups only).
"""
import argparse
import json
import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallcv import m1, pose, urfd  # noqa: E402
from fallcv.eval import evaluate, video_level  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--extractor", default="mediapipe", choices=pose.EXTRACTORS)
ap.add_argument("--win", type=float, default=1.0, help="window length in seconds")
args = ap.parse_args()

lab, onsets, groups = urfd.labels(), urfd.fall_onsets(), m1.load_groups()
X, Y, USE = m1.build(args.extractor, args.win, onsets)
gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                     class_weight="balanced", random_state=0)
merged, folds, _ = m1.loso(gbm, X, Y, USE, groups, onsets, lab)
per_seq, s = evaluate(merged, onsets, lab)

out = ROOT / "results"
out.mkdir(exist_ok=True)
tag = f"m1_{args.extractor}_w{args.win:g}"
folds.to_csv(out / f"{tag}_loso.csv", index=False)
per_seq.to_csv(out / f"{tag}_per_seq.csv", index=False)
keep = ["falls_hit", "event_recall", "late_detections", "median_latency_s", "adl_false_alarms",
        "adl_seqs_with_alarm", "pre_onset_false_alarms", "fa_per_hour_adl"]
(out / f"{tag}_summary.json").write_text(json.dumps(
    {"win_s": args.win, "recall_std": folds.recall.std(), **{k: s[k] for k in keep}, **video_level(merged)},
    indent=2, default=float))

print(f"M1 window features + GBM, extractor = {args.extractor}, window = {args.win:g} s")
print(folds.round(2).to_string(index=False))
print(f"pooled: falls {s['falls_hit']}, recall {folds.recall.mean():.2f} ± {folds.recall.std():.2f}, "
      f"ADL FA {s['adl_false_alarms']} ({s['fa_per_hour_adl']:.0f}/h), pre-onset FA {s['pre_onset_false_alarms']}, "
      f"latency {s['median_latency_s']:.2f} s")
print("video level:", video_level(merged))
print("missed falls:", ", ".join(per_seq[(per_seq.kind == "fall") & ~per_seq.hit.astype(bool)].seq) or "none")
print("ADLs with alarms:", ", ".join(per_seq[(per_seq.kind == "adl") & (per_seq.false_alarms > 0)].seq) or "none")
