"""Extract and cache poses for every URFD sequence.

    python scripts/extract_poses.py --extractor mediapipe
    python scripts/extract_poses.py --extractor yolo [--weights yolo11m-pose.pt]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fallcv import urfd  # noqa: E402
from fallcv.pose import EXTRACTORS, cache_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--extractor", choices=EXTRACTORS, required=True)
ap.add_argument("--weights", default=None, help="YOLO weights (default yolo11m-pose.pt)")
ap.add_argument("--seqs", nargs="*", help="subset of sequences (default: all)")
args = ap.parse_args()

for seq in args.seqs or urfd.sequences():
    out = cache_path(args.extractor, seq)
    if out.exists():
        continue
    # a fresh extractor per sequence: MediaPipe VIDEO mode tracks across frames
    ex = EXTRACTORS[args.extractor](**({"weights": args.weights} if args.weights else {}))
    t0 = time.time()
    kps, boxes = [], []
    for f in urfd.frames(seq):
        kp, box = ex(cv2.imread(str(f)))
        kps.append(kp); boxes.append(box)
    kps, boxes = np.array(kps, dtype=np.float32), np.array(boxes, dtype=np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kp=kps, box=boxes)
    n = len(kps)
    print(f"{seq}: {n} frames, detected {np.mean(~np.isnan(boxes[:, 0])):.0%}, {n / (time.time() - t0):.0f} fps",
          flush=True)
