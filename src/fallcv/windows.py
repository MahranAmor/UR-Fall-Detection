"""M1 features: causal sliding windows over per-frame pose signals.

Per frame, B0's state machine already measures torso angle, box aspect, height ratio (vs the person's own
upright height), vertical/horizontal speed and recent drops. Each frame t becomes one row summarising the
window [t - W + 1, t] (past frames only, so the detector stays causal). Missing values stay NaN; the GBM
handles them natively.

Two signal sets:
- "full": B0's signals + `cy_rel` (hip height in the image). Strong on URFD, but tied to the camera position.
- "view_free": no image-position signal. Instead, signals relative to a calibration of the person's own
  upright pose (first second the person is seen upright, like a 5 s calibration at the start of a demo):
  angle / aspect / size change vs that reference, size change speed and body-relative joint motion.

Training target per frame: 1 within [onset, onset + pos_s) of a fall, 0 for every ADL frame and every
fall frame before the onset. Fall frames after onset + pos_s are left out of training (the person is just
lying there; an alarm then is a late detection, not a false alarm).
"""
import numpy as np
import pandas as pd

from . import rule
from .urfd import FPS

BASE = ["angle", "aspect", "ratio", "v_down", "v_x", "d_ratio", "d_angle"]
SETS = {
    "full": BASE + ["cy_rel"],
    "view_free": BASE + ["d_angle_ref", "aspect_ref", "scale_ref", "v_scale", "motion"],
}
CALIB_FRAMES = FPS  # frames of upright pose used as the person's reference


def frame_signals(kp, box, p=rule.Params(), signals="full"):
    """Per-frame causal signals (one row per frame)."""
    _, _, feats = rule.run(kp, box, p)
    rows, ref, calib = [], None, []
    torso_prev, kp_prev = None, None
    diag_hist = []
    for i, f in enumerate(feats):
        r = {"seen": 0.0}
        if f is not None:
            r["seen"] = 1.0
            for k in BASE:
                v = f.get(k)
                r[k] = np.nan if v is None else float(v)
            r["cy_rel"] = f["cy"] / p.img_h
            # --- view-free extras ---
            x1, y1, x2, y2 = box[i, :4]
            diag = float(np.hypot(x2 - x1, y2 - y1)) if not np.isnan(x1) else np.nan
            if ref is None and f.get("posture") == "up" and not np.isnan(diag) and f.get("angle") is not None:
                calib.append((f["angle"], f["aspect"], diag))
                if len(calib) >= CALIB_FRAMES:
                    ref = np.nanmedian(np.array(calib, float), axis=0)
            if ref is not None:
                r["d_angle_ref"] = r["angle"] - ref[0]
                r["aspect_ref"] = r["aspect"] / ref[1] if r["aspect"] == r["aspect"] else np.nan
                r["scale_ref"] = diag / ref[2]
            diag_hist.append((i, diag))
            old = [d for j, d in diag_hist if i - j <= 9 and not np.isnan(d)]
            r["v_scale"] = (np.log(diag) - np.log(old[0])) * FPS / max(len(old) - 1, 1) \
                if len(old) > 1 and not np.isnan(diag) else np.nan
            # joint motion in torso lengths per second (body-relative, any viewpoint)
            sh, hp = np.nanmean(kp[i, [5, 6], :2], 0), np.nanmean(kp[i, [11, 12], :2], 0)
            torso = float(np.linalg.norm(sh - hp))
            if kp_prev is not None and torso == torso and torso > 5:
                d = np.linalg.norm(kp[i, :, :2] - kp_prev, axis=1)
                r["motion"] = float(np.nanmedian(d)) * FPS / torso if np.any(~np.isnan(d)) else np.nan
            kp_prev = kp[i, :, :2]
        rows.append(r)
    cols = SETS[signals] + ["seen"]
    return pd.DataFrame(rows).reindex(columns=cols).astype(float)


def window_features(sig, win_s=1.0):
    """Summary stats of each signal over the past `win_s` seconds, plus the current value."""
    w = max(int(round(win_s * FPS)), 2)
    names = [c for c in sig.columns if c != "seen"]
    roll = sig[names].rolling(w, min_periods=1)
    out = {"seen_frac": sig.seen.rolling(w, min_periods=1).mean()}
    for k in names:
        out[f"{k}_now"] = sig[k]
        out[f"{k}_min"] = roll[k].min()
        out[f"{k}_max"] = roll[k].max()
        out[f"{k}_mean"] = roll[k].mean()
        out[f"{k}_delta"] = sig[k] - sig[k].shift(w - 1)
    return pd.DataFrame(out)


def augment(kp, box, rng, img_w=640, img_h=480):
    """Simulate another camera: random rotation, scale, vertical squash (tilt) and shift of the 2D skeleton.

    One random transform per sequence (a fixed camera), applied to every frame's joints and box."""
    ang = np.radians(rng.uniform(-20, 20))
    s = rng.uniform(0.7, 1.2)
    sy = rng.uniform(0.6, 1.0)                       # looking down from higher up squashes vertical extents
    c = np.array([img_w / 2, img_h / 2])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.diag([s, s * sy])
    t = c + rng.uniform(-0.1, 0.1, 2) * [img_w, img_h]
    kp2 = kp.copy()
    kp2[..., :2] = (kp[..., :2] - c) @ R.T + t
    corners = np.stack([box[:, [0, 1]], box[:, [2, 1]], box[:, [0, 3]], box[:, [2, 3]]], 1)  # (n, 4, 2)
    cc = (corners - c) @ R.T + t
    box2 = box.copy()
    box2[:, :2], box2[:, 2:4] = cc.min(1), cc.max(1)
    return kp2, box2


def targets(seq, n, onsets, pos_s=3.0):
    """Per-frame target (1/0) and a train mask, frame numbers are index + 1."""
    y = np.zeros(n, dtype=int)
    use = np.ones(n, dtype=bool)
    if seq.startswith("fall"):
        on = onsets[seq] - 1
        end = min(n, on + int(pos_s * FPS))
        y[on:end] = 1
        use[end:] = False
    return y, use


def alarms_from_proba(proba, thr, k, refractory_s=3.0):
    """Alarm (1-based frame) when proba > thr for k consecutive frames; then stay silent for refractory_s."""
    out, run, quiet_until = [], 0, -1
    for i, q in enumerate(proba):
        run = run + 1 if q > thr else 0
        if run >= k and i > quiet_until:
            out.append(i + 1)
            quiet_until = i + int(refractory_s * FPS)
            run = 0
    return out
