"""Render a sequence with skeleton, state and features overlaid (for debugging and demos).

    python scripts/render_b0.py fall-01 --extractor yolo   ->  results/videos/fall-01_yolo.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallcv import pose, rule, urfd  # noqa: E402

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = {"STANDING": (80, 200, 80), "WALKING": (80, 200, 80), "SITTING": (200, 160, 60), "UNKNOWN": (150, 150, 150),
          "FALLING": (0, 165, 255), "LYING": (0, 220, 255), "FALL_DETECTED": (40, 40, 230)}

ap = argparse.ArgumentParser()
ap.add_argument("seq")
ap.add_argument("--extractor", default="mediapipe", choices=pose.EXTRACTORS)
args = ap.parse_args()

kp, box = pose.load(args.extractor, args.seq)
states, alarms, feats = rule.run(kp, box)
lab = urfd.labels().query("seq == @args.seq").set_index("frame").label
out = ROOT / "results" / "videos" / f"{args.seq}_{args.extractor}.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
vw = None
for i, f in enumerate(urfd.frames(args.seq)):
    img = cv2.imread(str(f))
    if vw is None:
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), urfd.FPS, (img.shape[1], img.shape[0]))
    s, k, b, ft = states[i], kp[i], box[i], feats[i]
    c = COLORS[s]
    if not np.isnan(b[0]):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
    for a, z in EDGES:
        if k[a, 2] > 0.3 and k[z, 2] > 0.3:
            cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[z, :2].astype(int)), (255, 255, 255), 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(img, s, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
    gt = {-1: "not lying", 0: "transition", 1: "lying"}.get(lab.get(i + 1), "-")
    cv2.putText(img, f"frame {i + 1}  gt: {gt}", (360, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if ft:
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        txt = f"angle {fmt(ft['angle'])}  aspect {fmt(ft['aspect'])}  h/ref {fmt(ft['ratio'])}  v_down {ft['v_down']:.2f}"
        cv2.putText(img, txt, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    if s == "FALL_DETECTED":
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), c, 6)
    vw.write(img)
vw.release()
print(f"{out}  alarms at frames {alarms}")
