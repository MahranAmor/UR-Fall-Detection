"""Step 3: MediaPipe skeletons for every RGB frame.

Usage:
  python scripts/extract_skeletons.py --compare   # part A: score settings on the hardest sequences
  python scripts/extract_skeletons.py             # part B: run the chosen setting on all 70 sequences

Part A writes results/skeleton_settings_comparison.csv (+ per-frame rows in skeleton_settings_frames.csv).
Part B writes data/processed/skeletons/<seq>.npz (git-ignored; re-runs skip finished sequences)
and results/skeleton_detection.csv. Each npz holds, for T frames:
  frame (T,)            frame number, the same numbering as the label csvs
  xy (T, 33, 2)         joint positions, normalised to [0, 1] of image width / height
  z (T, 33)             MediaPipe's relative depth per joint (scale differs between full-frame and crop poses)
  visibility (T, 33)    MediaPipe's confidence that the joint is visible
  detected (T,)         a skeleton was kept for this frame
  source (T,)           0 none, 1 full frame, 2 full frame rotated, 3 person crop, 4 person crop rotated
  rotated (T,)          the pose came from a 90-degree-rotated image (source 2 or 4)
  box (T, 4)            person box x, y, w, h normalised to the image (NaN if none; person_crop only)
Frames without a skeleton hold NaN, never 0, so they cannot pass for real positions.

Why a person detector: on fall-01 / fall-15 the full-frame pose model put skeletons on an office
chair next to the person lying on the floor, in every setting. A COCO person detector boxes the
lying person (98.6% of fall lying frames) and not the chair, so poses are estimated inside that box.
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "urfd"
POSE_MODEL = REPO / "models" / "pose_landmarker_full.task"
PERSON_MODEL = REPO / "models" / "efficientdet_lite0.tflite"
OUT = REPO / "data" / "processed" / "skeletons"
RESULTS = REPO / "results"
FPS = 30
W, H = 640, 480
N_JOINTS = 33

PERSON_SCORE = 0.2    # minimum person-box score
CROP_MARGIN = 0.25    # box grown by this fraction of its size on every side
TRUST_VIS = 0.8       # no box: keep a full-frame pose this confident (chair poses scored 0.52-0.62)
MAX_JUMP = 100 / W    # no box: or one within this distance (normalised) ...
MAX_GAP = 10          # ... of the last kept pose, at most this many frames ago

SETTINGS = {
    "image_c50": dict(mode="image", conf=0.5, rotate=False, person=False),
    "video_c50": dict(mode="video", conf=0.5, rotate=False, person=False),
    "video_c30": dict(mode="video", conf=0.3, rotate=False, person=False),
    "video_c30_rot": dict(mode="video", conf=0.3, rotate=True, person=False),
    "person_crop": dict(mode="video", conf=0.3, rotate=True, person=True),
}
CHOSEN = "person_crop"  # set from the part A comparison
HARD = ["adl-23", "adl-38", "adl-22", "adl-19", "fall-01", "fall-15", "fall-28"]


def list_frames(seq):
    """(frame number, path) sorted by the number in the file name."""
    out = []
    for p in (RAW / "rgb" / seq).rglob("*.png"):
        m = re.search(r"-(\d+)\.png$", p.name)
        out.append((int(m.group(1)), p))
    return sorted(out)


def landmarker(mode, conf):
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=conf,
        min_pose_presence_confidence=conf, min_tracking_confidence=conf)
    return vision.PoseLandmarker.create_from_options(opts)


def person_detector():
    opts = vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(PERSON_MODEL)),
        running_mode=vision.RunningMode.IMAGE, score_threshold=PERSON_SCORE,
        max_results=3, category_allowlist=["person"])
    return vision.ObjectDetector.create_from_options(opts)


def as_image(rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def to_arrays(lms):
    a = np.array([[p.x, p.y, p.z, p.visibility] for p in lms], dtype=np.float32)
    return a[:, :2], a[:, 2], a[:, 3]


def rotated_pose(det_img, rgb):
    """Try the image turned 90 degrees both ways; map the better pose back to the unrotated image.

    A lying person looks upright after a quarter turn, which is what pose models are trained on.
    For ROTATE_90_CLOCKWISE a point (u, v) in the rotated image is (v, 1 - u) in the original;
    for ROTATE_90_COUNTERCLOCKWISE it is (1 - v, u). Coordinates are normalised, so size does not matter.
    Checked: on upright frames the mapped pose lands ~1 px from the direct pose.
    """
    best = None
    for rot, back in [(cv2.ROTATE_90_CLOCKWISE, lambda u, v: (v, 1 - u)),
                      (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda u, v: (1 - v, u))]:
        res = det_img.detect(as_image(cv2.rotate(rgb, rot)))
        if res.pose_landmarks:
            xy, z, vis = to_arrays(res.pose_landmarks[0])
            x, y = back(xy[:, 0], xy[:, 1])
            cand = (np.stack([x, y], 1), z, vis)
            if best is None or vis.mean() > best[2].mean():
                best = cand
    return best


def person_boxes(od, rgb):
    """Person boxes as (x, y, w, h) in pixels, best score first."""
    dets = od.detect(as_image(rgb)).detections
    dets = sorted(dets, key=lambda d: -d.categories[0].score)
    return [(d.bounding_box.origin_x, d.bounding_box.origin_y, d.bounding_box.width, d.bounding_box.height)
            for d in dets]


def pick_box(boxes, last_c):
    """The box nearest the last kept pose (same person), else the highest-scoring one."""
    if last_c is None:
        return boxes[0]
    centre = lambda b: np.array([(b[0] + b[2] / 2) / W, (b[1] + b[3] / 2) / H])
    return min(boxes, key=lambda b: np.linalg.norm(centre(b) - last_c))


def crop_pose(det_img, rgb, box, rotate):
    """Pose inside the grown person box, mapped back to full-image coordinates. Returns (pose, source)."""
    x, y, w, h = box
    x0, y0 = max(0, int(x - CROP_MARGIN * w)), max(0, int(y - CROP_MARGIN * h))
    x1, y1 = min(W, int(x + w + CROP_MARGIN * w)), min(H, int(y + h + CROP_MARGIN * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, 0
    crop = rgb[y0:y1, x0:x1]
    res = det_img.detect(as_image(crop))
    pose, src = None, 0
    if res.pose_landmarks:
        pose, src = to_arrays(res.pose_landmarks[0]), 3
    elif rotate:
        pose, src = rotated_pose(det_img, crop), 4
    if pose is None:
        return None, 0
    xy, z, vis = pose
    xy = np.stack([(x0 + xy[:, 0] * (x1 - x0)) / W, (y0 + xy[:, 1] * (y1 - y0)) / H], 1).astype(np.float32)
    return (xy, z, vis), src


def centre(xy):
    return np.nanmedian(xy, 0)



def extract(seq, mode, conf, rotate, person=False):
    frames = list_frames(seq)
    T = len(frames)
    xy = np.full((T, N_JOINTS, 2), np.nan, np.float32)
    z = np.full((T, N_JOINTS), np.nan, np.float32)
    vis = np.full((T, N_JOINTS), np.nan, np.float32)
    source = np.zeros(T, np.int8)
    boxes = np.full((T, 4), np.nan, np.float32)
    # a fresh landmarker per sequence: VIDEO mode tracks across frames and needs increasing timestamps
    det = landmarker(mode, conf)
    det_img = landmarker("image", conf) if (rotate or person) else None
    od = person_detector() if person else None
    last_c, last_i = None, -10 ** 6
    try:
        for i, (fno, path) in enumerate(frames):
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            img = as_image(rgb)
            res = det.detect_for_video(img, int(fno * 1000 / FPS)) if mode == "video" else det.detect(img)
            full = to_arrays(res.pose_landmarks[0]) if res.pose_landmarks else None
            pose, src = None, 0
            if not person:
                if full is not None:
                    pose, src = full, 1
                elif rotate:
                    pose = rotated_pose(det_img, rgb)
                    src = 2 if pose is not None else 0
            else:
                bxs = person_boxes(od, rgb)
                if bxs:
                    b = pick_box(bxs, last_c)
                    boxes[i] = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]
                    pose, src = crop_pose(det_img, rgb, b, rotate)
                    # crop failed: a full-frame pose only if very confident. "Centre inside the box" is not
                    # enough: on fall-15 f66 a chair at the box edge passed that test (visibility 0.67).
                    if pose is None and full is not None and full[2].mean() >= TRUST_VIS:
                        pose, src = full, 1
                elif full is not None:
                    near = last_c is not None and i - last_i <= MAX_GAP and \
                        np.linalg.norm(centre(full[0]) - last_c) <= MAX_JUMP
                    if full[2].mean() >= TRUST_VIS or near:
                        pose, src = full, 1
            if pose is not None:
                xy[i], z[i], vis[i] = pose
                source[i] = src
                last_c, last_i = centre(xy[i]), i
    finally:
        for m in (det, det_img, od):
            if m is not None:
                m.close()
    return dict(frame=np.array([f for f, _ in frames]), xy=xy, z=z, visibility=vis,
                detected=source > 0, source=source, rotated=np.isin(source, [2, 4]), box=boxes)


def frame_labels():
    lab = pd.concat([pd.read_csv(RAW / "labels" / f"urfall-cam0-{k}.csv", header=None, usecols=[0, 1, 2])
                     for k in ["falls", "adls"]], ignore_index=True)
    lab.columns = ["seq", "frame", "label"]
    return lab.set_index(["seq", "frame"]).label


def detection_table(seq, sk, labels, setting):
    phase = {-1: "standing", 0: "transition", 1: "lying"}
    lab = [labels.get((seq, f)) for f in sk["frame"]]
    with np.errstate(invalid="ignore"):
        mean_vis = np.nanmean(np.where(sk["detected"][:, None], sk["visibility"], np.nan), 1) \
            if sk["detected"].any() else np.full(len(lab), np.nan)
    return pd.DataFrame({"setting": setting, "seq": seq, "frame": sk["frame"],
                         "phase": [phase.get(l, "unlabeled") if l is not None else "unlabeled" for l in lab],
                         "detected": sk["detected"], "source": sk["source"], "rotated": sk["rotated"],
                         "mean_vis": mean_vis})


def compare(names):
    labels = frame_labels()
    rows = []
    for name in names:
        t0 = time.time()
        for seq in HARD:
            rows.append(detection_table(seq, extract(seq, **SETTINGS[name]), labels, name))
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d["kind"] = d.seq.str.split("-").str[0]
    RESULTS.mkdir(exist_ok=True)
    d.to_csv(RESULTS / "skeleton_settings_frames.csv", index=False, lineterminator="\n", float_format="%.3f")
    tab = d.pivot_table(index="setting", columns=["kind", "phase"], values="detected", aggfunc="mean")
    tab = tab.reindex(names).round(3)
    tab[("all", "all")] = d.groupby("setting").detected.mean().reindex(names).round(3)
    tab.to_csv(RESULTS / "skeleton_settings_comparison.csv", lineterminator="\n")
    pd.set_option("display.width", 200)
    print("\nDetection rate on the hardest sequences, by setting and phase:\n")
    print(tab.to_string())


def run_all():
    labels = frame_labels()
    cfg = SETTINGS[CHOSEN]
    OUT.mkdir(parents=True, exist_ok=True)
    seqs = sorted(p.name for p in (RAW / "rgb").iterdir() if p.is_dir())
    rows = []
    for i, seq in enumerate(seqs, 1):
        f = OUT / f"{seq}.npz"
        if f.exists():
            sk = dict(np.load(f))
        else:
            sk = extract(seq, **cfg)
            np.savez_compressed(f, **sk)
        rows.append(detection_table(seq, sk, labels, CHOSEN))
        print(f"[{i}/{len(seqs)}] {seq}: {sk['detected'].mean():.0%} detected", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d["kind"] = d.seq.str.split("-").str[0]
    per_seq = d.pivot_table(index="seq", columns="phase", values="detected", aggfunc="mean").round(3)
    per_seq["all"] = d.groupby("seq").detected.mean().round(3)
    per_seq["crop_share"] = d.groupby("seq").source.apply(lambda s: s.isin([3, 4]).mean()).round(3)
    RESULTS.mkdir(exist_ok=True)
    per_seq.to_csv(RESULTS / "skeleton_detection.csv", lineterminator="\n")
    print("\nDetection rate by type and phase:\n")
    print(d.pivot_table(index="kind", columns="phase", values="detected", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="score the settings on the hardest sequences")
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    args = ap.parse_args()
    compare(args.settings) if args.compare else run_all()
