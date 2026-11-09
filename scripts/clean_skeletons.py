"""Step 4: clean the raw skeletons from step 3.

Usage: python scripts/clean_skeletons.py [--smooth N]

Reads data/processed/skeletons/<seq>.npz, writes data/processed/clean/<seq>.npz (git-ignored)
and results/clean_summary.csv. Per sequence, in this order:

0. Remove flips: frames whose hip midpoint is more than half a torso length from the median of the
   7 frames around them become missing. For folded or curled-up bodies the pose model sometimes fits the
   skeleton upside down for a few frames; real motion, even a fall, does not trigger this.
1. Remove guesses: joints with visibility < VIS_MIN become missing. Only clear guesses are removed:
   during transitions and lying about half of all leg joints have visibility < 0.5 (hidden, but their
   positions are still usable), so a strict cut-off would delete the legs from most fall frames.
   The visibility value is kept for every joint so the models can weigh it.
2. Fill short gaps: per joint, linear interpolation across gaps of at most MAX_FILL frames
   (93% of the gaps between detected frames). Longer gaps and gaps at the start or end stay missing.
3. Smooth: Savitzky-Golay filter (window SMOOTH, polynomial 2) on each unbroken run of a joint;
   it removes jitter but keeps sharp peaks, such as the fall, better than a moving average.

Positions stay in pixels. Centring and scaling happen per clip in step 5, relative to the clip's first
frames, because a sequence-level normalisation does not work here:
- one scale per sequence: people walk towards / away from the camera, so their size changes 1.8x within
  a typical ADL (1.1x in falls), which distorts ADLs far more than falls;
- a scale tracked over time from "upright" frames: in 2D, a person lying along the camera axis still has
  the shoulders above the hips, so lying frames pass as upright and the fall's drop disappears.
Within a 1.5 s clip, the size change from walking is small, and a fall clip starts before the fall.

Output fields, for T frames:
  frame (T,)          frame number (label numbering)
  xy (T, 33, 2)       joint positions in pixels (640x480 image; x right, y down); NaN where missing
  visibility (T, 33)  MediaPipe visibility (NaN where missing)
  state (T, 33)       0 missing, 1 observed, 2 filled
  torso_px (T,)       shoulder-midpoint to hip-midpoint distance in pixels (NaN where missing)
  spike (T,)          frame removed by the flip filter (before gap filling)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parent.parent
IN = REPO / "data" / "processed" / "skeletons"
OUT = REPO / "data" / "processed" / "clean"
RESULTS = REPO / "results"
PX = np.array([640.0, 480.0])
VIS_MIN = 0.1
MAX_FILL = 5
SPIKE_HALF = 3       # spike filter window: 3 frames each side
SPIKE_TORSO = 0.5    # a real fall moves the hips at most ~0.19 torso lengths per frame
# Savitzky-Golay window in frames (odd), 0 = off. Tested on all sequences: window 5 cuts standing jitter
# to 40% and keeps 94% (worst 66%) of each fall's peak 0.17 s hip speed; window 7 cuts jitter to 24%
# but keeps 88% (worst 53%). The fall's speed matters more than residual jitter, so 5.
SMOOTH = 5
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24


def runs(mask):
    """(start, end) index pairs of runs of True, end exclusive."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fill_short_gaps(x, max_len):
    """Linear interpolation over interior NaN gaps of at most max_len; returns filled copy and filled mask."""
    x = x.copy()
    filled = np.zeros(len(x), bool)
    for s, e in runs(np.isnan(x)):
        if s == 0 or e == len(x) or e - s > max_len:
            continue
        x[s:e] = np.interp(np.arange(s, e), [s - 1, e], [x[s - 1], x[e]])
        filled[s:e] = True
    return x, filled


def smooth(x, window):
    """Savitzky-Golay on each unbroken run; runs shorter than the window are left as they are."""
    if window < 3:
        return x
    x = x.copy()
    for s, e in runs(~np.isnan(x)):
        if e - s >= window:
            x[s:e] = savgol_filter(x[s:e], window, 2)
    return x


def spikes(xy, detected):
    """Frames whose hip midpoint is > SPIKE_TORSO torso lengths from the median of the frames around them.

    For a folded or curled-up body the pose model sometimes fits the skeleton upside down for a few
    frames (hips placed at the head), which shows up as impossible hip jumps. The window median follows
    the majority orientation, so brief flips stand out, while steady motion, even a fast fall, keeps the
    centre frame close to its window median and is not touched.
    """
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    torso = np.linalg.norm((xy[:, L_SH] + xy[:, R_SH]) / 2 - hip, axis=1)
    roll = lambda a: pd.Series(a).rolling(2 * SPIKE_HALF + 1, center=True, min_periods=3).median().values
    med = np.stack([roll(hip[:, 0]), roll(hip[:, 1])], 1)
    with np.errstate(invalid="ignore"):
        return detected & (np.linalg.norm(hip - med, axis=1) > SPIKE_TORSO * roll(torso))


def clean(raw, smooth_window=SMOOTH):
    xy = raw["xy"].astype(np.float64) * PX                       # pixels
    vis = raw["visibility"].astype(np.float64)
    J = xy.shape[1]
    removed = spikes(xy, raw["detected"])
    observed = (raw["detected"] & ~removed)[:, None] & (vis >= VIS_MIN)
    xy[~observed] = np.nan
    vis[~observed] = np.nan

    state = observed.astype(np.int8)
    for j in range(J):
        for c in range(2):
            xy[:, j, c], f = fill_short_gaps(xy[:, j, c], MAX_FILL)
            if c == 0:
                state[f, j] = 2
        vis[:, j], _ = fill_short_gaps(vis[:, j], MAX_FILL)
        for c in range(2):
            xy[:, j, c] = smooth(xy[:, j, c], smooth_window)

    torso = np.linalg.norm((xy[:, L_SH] + xy[:, R_SH]) / 2 - (xy[:, L_HIP] + xy[:, R_HIP]) / 2, axis=1)
    return dict(frame=raw["frame"], xy=xy.astype(np.float32), visibility=vis.astype(np.float32),
                state=state, torso_px=torso.astype(np.float32), spike=removed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=int, default=SMOOTH, help="Savitzky-Golay window (odd), 0 = off")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(IN.glob("*.npz")):
        raw = np.load(f)
        c = clean(raw, args.smooth)
        np.savez_compressed(OUT / f.name, **c)
        body_ok = ~np.isnan(c["xy"][:, [L_SH, R_SH, L_HIP, R_HIP], 0]).any(1)
        rows.append({"seq": f.stem, "frames": len(c["frame"]), "raw_detected": raw["detected"].mean(),
                     "usable_after_fill": body_ok.mean(), "joints_filled": (c["state"] == 2).mean(),
                     "joints_missing": (c["state"] == 0).mean(), "spike_frames": c["spike"].mean()})
    s = pd.DataFrame(rows).set_index("seq").round(3)
    RESULTS.mkdir(exist_ok=True)
    s.to_csv(RESULTS / "clean_summary.csv", lineterminator="\n")
    s["kind"] = s.index.str.split("-").str[0]
    print(s.groupby("kind")[["raw_detected", "spike_frames", "usable_after_fill", "joints_filled",
                             "joints_missing"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
