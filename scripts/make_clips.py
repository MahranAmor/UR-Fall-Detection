"""Step 5: cut the cleaned skeletons into fixed-length clips and normalise each clip by its own start.

Usage: python scripts/make_clips.py [--min-usable 0.7]

Reads data/processed/clean/<seq>.npz and the frame labels. Writes
  data/processed/clips.npz    arrays for the models (git-ignored)
  results/clips_index.csv     one row per clip: sequence, start frame, label, checks

Clips: WINDOW frames (1.5 s), a new clip every STRIDE frames, every frame labeled.
Labels (every fall's transition is exactly 30 contiguous frames):
  fall (1)      fall sequence, clip holds >= MIN_TRANSITION transition frames (at least 0.5 s of the fall)
  not fall (0)  every ADL clip (including lying down on purpose), and fall-sequence clips before the fall
  left out      fall-sequence clips with 1..MIN_TRANSITION-1 transition frames (only the edge of the fall)
                or after the fall only (already lying: no fall happens in the clip)

Normalisation per clip (not per sequence: body size changes 1.8x within a typical ADL as people walk):
  reference = the first REF_FRAMES usable frames (hips and shoulders present) of the clip
  pose      joints minus that frame's hip midpoint, divided by the reference torso length (body shape)
  traj      hip midpoint minus the reference hip midpoint, same units; y > 0 means lower (the drop)
Remaining gaps inside a clip are filled per joint (linear, nearest at the edges); a joint never seen in
the clip sits at the hip centre with visibility 0.

Arrays in clips.npz, N clips:
  pose (N, 45, 33, 2), traj (N, 45, 2), visibility (N, 45, 33; 0 where filled), label (N,),
  usable (N, 45) frames observed before in-clip filling. `usable` is for checks only, not a model input:
  missing frames are more common in falls (flip removal), so they would be a shortcut.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LABELS = REPO / "data" / "raw" / "urfd" / "labels"
IN = REPO / "data" / "processed" / "clean"
OUT = REPO / "data" / "processed" / "clips.npz"
INDEX = REPO / "results" / "clips_index.csv"
WINDOW = 45
STRIDE = 15
MIN_TRANSITION = 15
REF_FRAMES = 10
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24


def frame_labels():
    lab = pd.concat([pd.read_csv(LABELS / f"urfall-cam0-{k}.csv", header=None, usecols=[0, 1, 2])
                     for k in ["falls", "adls"]], ignore_index=True)
    lab.columns = ["seq", "frame", "label"]
    return lab


def fill_in_clip(a):
    """Fill NaN along time (axis 0) per column: linear inside, nearest at the edges. All-NaN columns stay NaN."""
    a = a.copy()
    t = np.arange(len(a))
    flat = a.reshape(len(a), -1)
    for k in range(flat.shape[1]):
        ok = ~np.isnan(flat[:, k])
        if 0 < ok.sum() < len(t):
            flat[:, k] = np.interp(t, t[ok], flat[ok, k])
    return flat.reshape(a.shape)


def clip_label(is_fall, lab):
    if not is_fall:
        return 0
    n_tr = int((lab == 0).sum())
    if n_tr >= MIN_TRANSITION:
        return 1
    if n_tr == 0 and (lab == -1).all():
        return 0
    return -1  # left out


def normalise(xy, vis):
    """xy (W, 33, 2) pixels with NaN; returns pose, traj, visibility, usable mask, scale (px) or None."""
    usable = ~np.isnan(xy[:, [L_SH, R_SH, L_HIP, R_HIP], 0]).any(1)
    if usable.sum() < 3:
        return None
    ref = np.where(usable)[0][:REF_FRAMES]
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    torso = np.linalg.norm((xy[:, L_SH] + xy[:, R_SH]) / 2 - hip, axis=1)
    scale = float(np.median(torso[ref]))
    hip_ref = np.median(hip[ref], 0)

    xy_f = fill_in_clip(xy)
    v = np.where(np.isnan(xy[..., 0]), 0.0, np.nan_to_num(vis, nan=0.0))   # filled -> visibility 0
    hip_f = (xy_f[:, L_HIP] + xy_f[:, R_HIP]) / 2
    pose = (xy_f - hip_f[:, None, :]) / scale
    pose = np.nan_to_num(pose, nan=0.0)                                    # joint never seen: at the hip
    traj = (hip_f - hip_ref) / scale
    return pose, traj, v, usable, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-usable", type=float, default=0.7,
                    help="keep clips with at least this share of usable frames; 0.7 keeps 70 of 71 fall clips and fills at most 13 of 45 frames. A model given only missing-frame information scores AUC 0.53 at 0.7 (chance), so this is a quality choice, not a shortcut fix")
    args = ap.parse_args()
    lab = frame_labels()
    poses, trajs, viss, usables, rows = [], [], [], [], []
    for f in sorted(IN.glob("*.npz")):
        s = f.stem
        d = np.load(f)
        fr = d["frame"]
        lab_s = lab[lab.seq == s].set_index("frame").label
        idx = {x: i for i, x in enumerate(fr)}
        for st in range(lab_s.index.min(), lab_s.index.max() - WINDOW + 2, STRIDE):
            frames = np.arange(st, st + WINDOW)
            win = lab_s.reindex(frames)
            if win.isna().any() or any(x not in idx for x in frames):
                continue
            y = clip_label(s.startswith("fall"), win.values)
            if y < 0:
                continue
            ii = [idx[x] for x in frames]
            n = normalise(d["xy"][ii].astype(np.float64), d["visibility"][ii].astype(np.float64))
            if n is None:
                rows.append(dict(seq=s, start=st, label=y, kept=False, reason="too few usable frames"))
                continue
            pose, traj, v, usable, scale = n
            frac = float(usable.mean())
            row = dict(seq=s, start=st, label=y, kept=frac >= args.min_usable,
                       reason="" if frac >= args.min_usable else f"usable < {args.min_usable}",
                       usable_frac=round(frac, 3), filled_joint_frac=round(float((v == 0).mean()), 3),
                       transition_frames=int((win == 0).sum()), lying_frames=int((win == 1).sum()),
                       scale_px=round(scale, 1), max_drop=round(float(traj[:, 1].max()), 3))
            rows.append(row)
            if row["kept"]:
                poses.append(pose); trajs.append(traj); viss.append(v); usables.append(usable)
    idx_df = pd.DataFrame(rows)
    kept = idx_df[idx_df.kept].reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, pose=np.array(poses, np.float32), traj=np.array(trajs, np.float32),
                        visibility=np.array(viss, np.float32), usable=np.array(usables),
                        label=kept.label.to_numpy(np.int8), seq=np.array(kept.seq.tolist(), dtype="U16"),
                        start=kept.start.to_numpy(np.int32))
    INDEX.parent.mkdir(exist_ok=True)
    idx_df.to_csv(INDEX, index=False, lineterminator="\n")
    print(f"clips kept: {len(kept)} (fall {int(kept.label.sum())}, not fall {int((kept.label == 0).sum())}); "
          f"dropped for missing frames: {int((~idx_df.kept).sum())}")
    print(idx_df[~idx_df.kept].groupby("label").size().rename("dropped by label").to_string())


if __name__ == "__main__":
    main()
