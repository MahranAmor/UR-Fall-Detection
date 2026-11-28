"""Live B0 demo: webcam (or a video file) -> pose -> causal state machine, drawn on screen.

    python scripts/live_demo.py                      # default webcam, MediaPipe
    python scripts/live_demo.py --cam 1              # another camera
    python scripts/live_demo.py --video clip.mp4     # a video file instead
    python scripts/live_demo.py --extractor yolo     # YOLO-pose (needs torch + ultralytics)

Keys: q quit, r reset (stand fully in view, head to feet, so it learns your upright height).
Tips: camera about 2-3 m away at chest height, whole body visible, good light, fall onto a mattress.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fallcv import pose, rule  # noqa: E402

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = {"STANDING": (80, 200, 80), "WALKING": (80, 200, 80), "SITTING": (200, 160, 60), "UNKNOWN": (150, 150, 150),
          "FALLING": (0, 165, 255), "LYING": (0, 220, 255), "FALL_DETECTED": (40, 40, 230)}

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--video")
ap.add_argument("--extractor", default="mediapipe", choices=pose.EXTRACTORS)
ap.add_argument("--fps", type=float, help="force the fps used by the timers (default: measured)")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video if args.video else args.cam)
if not cap.isOpened():
    sys.exit(f"cannot open {'video ' + args.video if args.video else f'camera {args.cam}'}")
det = pose.EXTRACTORS[args.extractor]()


def read():
    ok, img = cap.read()
    return cv2.resize(img, (640, 480)) if ok else None  # thresholds were tuned on 640x480


# The timers count frames, so measure the real processing speed first (pose is the slow part).
if args.fps:
    fps = args.fps
elif args.video:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
else:
    t0, n = time.time(), 0
    while n < 30:
        img = read()
        if img is None:
            sys.exit("camera stopped")
        det(img)
        n += 1
    fps = n / (time.time() - t0)
fps = max(5, min(30, round(fps)))
print(f"running at {fps} fps (timers use this)")


def new_sm():
    return rule.FallStateMachine(rule.Params(fps=fps))


sm, n_alarms, flash = new_sm(), 0, 0
while True:
    img = read()
    if img is None:
        break
    kp, box = det(img)
    s, alarm, ft = sm.step(kp, box)
    if alarm:
        n_alarms += 1
        flash = fps * 2
        print(f"FALL DETECTED ({time.strftime('%H:%M:%S')})")
    c = COLORS[s]
    if not np.isnan(box[0]):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), c, 2)
    for a, z in EDGES:
        if kp[a, 2] > 0.3 and kp[z, 2] > 0.3:
            cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[z, :2].astype(int)), (255, 255, 255), 2)
    cv2.rectangle(img, (0, 0), (640, 58), (0, 0, 0), -1)
    cv2.putText(img, s, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    cv2.putText(img, f"alarms {n_alarms}   {fps} fps   r reset  q quit", (300, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    if sm.h_ref is None:
        cv2.putText(img, "stand fully in view (head to feet) to start", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    elif ft and ft.get("ratio") is not None:
        cv2.putText(img, f"angle {ft['angle'] or 0:.0f}  h/ref {ft['ratio']:.2f}  v_down {ft['v_down']:.2f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    if flash > 0:
        flash -= 1
        cv2.rectangle(img, (0, 0), (639, 479), COLORS["FALL_DETECTED"], 10)
        cv2.putText(img, "FALL!", (230, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.5, COLORS["FALL_DETECTED"], 6)
    cv2.imshow("Fall detection (B0)", img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("r"):
        sm, flash = new_sm(), 0
cap.release()
cv2.destroyAllWindows()
