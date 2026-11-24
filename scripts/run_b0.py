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
