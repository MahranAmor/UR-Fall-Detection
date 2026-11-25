"""Event-level evaluation of fall alarms on URFD.

Fall sequence: onset = first transition frame (label 0). A hit is the first alarm in [onset, onset + window].
Alarms before the onset, and every alarm in an ADL sequence, are false alarms.
Frame-level 'down' agreement: predicted LYING/FALL_DETECTED vs label 1 (label 0 frames excluded).
"""
import numpy as np
import pandas as pd

from .urfd import FPS


def evaluate(results, onsets, lab, window_s=3.0):
    """results: {seq: (states, alarm_frames)}. Returns (per-sequence table, summary dict)."""
    rows = []
    for seq, (states, alarms) in results.items():
        r = dict(seq=seq, kind=seq.split("-")[0], n_frames=len(states), n_alarms=len(alarms))
        if r["kind"] == "fall":
            on = onsets[seq]
            pre = [a for a in alarms if a < on]
            post = [a for a in alarms if a >= on]
            r.update(onset=on, false_alarms=len(pre),
                     hit=bool(post) and post[0] <= on + window_s * FPS,
                     detected_late=bool(post) and post[0] > on + window_s * FPS,
                     latency_s=(post[0] - on) / FPS if post else np.nan)
        else:
            r.update(false_alarms=len(alarms))
        rows.append(r)
    df = pd.DataFrame(rows)

    # frame-level 'down' vs label 1
    lab = lab[lab.label != 0]
    tp = fp = fn = 0
    for seq, (states, _) in results.items():
        d = lab[lab.seq == seq]
        pred = np.array([states[f - 1] in ("LYING", "FALL_DETECTED") for f in d.frame])
        true = d.label.values == 1
        tp += (pred & true).sum(); fp += (pred & ~true).sum(); fn += (~pred & true).sum()

    falls, adls = df[df.kind == "fall"], df[df.kind == "adl"]
    adl_hours = adls.n_frames.sum() / FPS / 3600
    summary = {
        "falls_hit": f"{int(falls.hit.sum())}/{len(falls)}",
        "event_recall": falls.hit.mean(),
        "late_detections": int(falls.detected_late.sum()),
        "median_latency_s": falls.latency_s.median(),
        "adl_false_alarms": int(adls.false_alarms.sum()),
        "adl_seqs_with_alarm": f"{int((adls.false_alarms > 0).sum())}/{len(adls)}",
        "pre_onset_false_alarms": int(falls.false_alarms.sum()),
        "fa_per_hour_adl": adls.false_alarms.sum() / adl_hours,
        "down_precision": tp / max(tp + fp, 1),
        "down_recall": tp / max(tp + fn, 1),
    }
    summary["down_f1"] = 2 * summary["down_precision"] * summary["down_recall"] / \
        max(summary["down_precision"] + summary["down_recall"], 1e-9)
    return df, summary


def video_level(results):
    """Offline score of notebooks 02-10: a video is 'fall' if it raised any alarm at all."""
    det = {s: len(a) > 0 for s, (_, a) in results.items()}
    falls = [s for s in det if s.startswith("fall")]
    adls = [s for s in det if s.startswith("adl")]
    caught = sum(det[s] for s in falls)
    fa = sum(det[s] for s in adls)
    return {"videos_right": f"{caught + len(adls) - fa}/{len(det)}", "video_accuracy": (caught + len(adls) - fa) / len(det),
            "falls_caught": f"{caught}/{len(falls)}", "adl_videos_alarmed": f"{fa}/{len(adls)}"}
