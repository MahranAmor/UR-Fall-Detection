"""B0 rule-based baseline: run the state machine on cached poses and report event-level metrics.

    python scripts/run_b0.py --extractor mediapipe
    python scripts/run_b0.py --extractor yolo --sweep

Thresholds are fixed by hand (no training), so every sequence is test data. --sweep shows sensitivity
to the two main thresholds; picking the best sweep row would be tuning on the test set.
"""
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallcv import pose, rule, urfd  # noqa: E402
from fallcv.eval import evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--extractor", default="mediapipe", choices=pose.EXTRACTORS)
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--loso", action="store_true")
args = ap.parse_args()

lab, onsets = urfd.labels(), urfd.fall_onsets()
data = {s: pose.load(args.extractor, s) for s in urfd.sequences()}
out = ROOT / "results"
out.mkdir(exist_ok=True)


def run_all(p):
    return {s: rule.run(kp, box, p)[:2] for s, (kp, box) in data.items()}


p = rule.Params()
per_seq, summary = evaluate(run_all(p), onsets, lab)
per_seq.to_csv(out / f"b0_{args.extractor}_per_seq.csv", index=False)
(out / f"b0_{args.extractor}_summary.json").write_text(json.dumps({"params": asdict(p), **summary}, indent=2,
                                                                  default=float))
print(f"B0 rule baseline, extractor = {args.extractor}")
for k, v in summary.items():
    print(f"  {k:24s} {v:.3f}" if isinstance(v, float) else f"  {k:24s} {v}")
print("\nmissed falls:", ", ".join(per_seq[(per_seq.kind == "fall") & ~per_seq.hit.astype(bool)].seq) or "none")
print("ADLs with alarms:", ", ".join(per_seq[(per_seq.kind == "adl") & (per_seq.false_alarms > 0)].seq) or "none")

if args.sweep:
    rows = []
    for drop in [0.3, 0.4, 0.5]:
        for confirm in [0.3, 0.5, 1.0]:
            _, s = evaluate(run_all(replace(p, drop_ratio=drop, confirm_s=confirm)), onsets, lab)
            rows.append(dict(drop_ratio=drop, confirm_s=confirm, **{k: s[k] for k in
                        ["falls_hit", "adl_false_alarms", "fa_per_hour_adl", "median_latency_s"]}))
    sw = pd.DataFrame(rows)
    sw.to_csv(out / f"b0_{args.extractor}_sweep.csv", index=False)
    print("\nsensitivity (in-sample, not for model selection):")
    print(sw.round(2).to_string(index=False))

if args.loso:
    # Leave-one-group-out (annotations/subjects.csv): pick the grid row on 4 groups, test on the 5th.
    groups = pd.read_csv(ROOT / "annotations" / "subjects.csv").set_index("seq").group
    grid = [replace(p, drop_ratio=d, confirm_s=c) for d in [0.3, 0.4, 0.5] for c in [0.3, 0.5, 1.0]]
    runs = [run_all(g) for g in grid]
    merged, rows = {}, []
    for g in sorted(groups.unique()):
        train = [s for s in data if groups[s] != g]
        score = lambda r: (lambda d: d.hit.fillna(False).astype(int).sum() - d.false_alarms.sum())(
            evaluate({s: r[s] for s in train}, onsets, lab)[0])
        best = max(range(len(grid)), key=lambda i: score(runs[i]))
        test = {s: runs[best][s] for s in data if groups[s] == g}
        merged.update(test)
        _, s = evaluate(test, onsets, lab)
        rows.append(dict(group=g, drop_ratio=grid[best].drop_ratio, confirm_s=grid[best].confirm_s,
                         falls_hit=s["falls_hit"], recall=s["event_recall"], adl_fa=s["adl_false_alarms"]))
    folds = pd.DataFrame(rows)
    _, s = evaluate(merged, onsets, lab)
    folds.to_csv(out / f"b0_{args.extractor}_loso.csv", index=False)
    print("\nLOSO (params picked on 4 groups, tested on the 5th):")
    print(folds.round(2).to_string(index=False))
    print(f"pooled: falls {s['falls_hit']}, recall {folds.recall.mean():.2f} ± {folds.recall.std():.2f}, "
          f"ADL FA {s['adl_false_alarms']} ({s['fa_per_hour_adl']:.0f}/h), latency {s['median_latency_s']:.2f} s")
