"""Extract YOLO poses on URFD cam1 (ceiling camera, falls only; synchronised with cam0, same frame numbers).

    python scripts/extract_poses_cam1.py [--extractor yolo]   ->  data/interim/poses/<extractor>_cam1/<seq>.npz
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
ap.add_argument("--extractor", default="yolo", choices=EXTRACTORS)
args = ap.parse_args()

for seq in [s for s in urfd.sequences() if s.startswith("fall")]:
    out = cache_path(f"{args.extractor}_cam1", seq)
    if out.exists():
        continue
    ex = EXTRACTORS[args.extractor]()
    t0, kps, boxes = time.time(), [], []
    for f in sorted((urfd.URFD / "rgb_cam1" / seq).rglob("*.png")):
        kp, box = ex(cv2.imread(str(f)))
        kps.append(kp); boxes.append(box)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, kp=np.array(kps, np.float32), box=np.array(boxes, np.float32))
    print(f"{seq}: {len(kps)} frames, detected {np.mean(~np.isnan(np.array(boxes)[:, 0])):.0%}, "
          f"{len(kps) / (time.time() - t0):.0f} fps", flush=True)
