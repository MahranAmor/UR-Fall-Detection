"""B0: rule-based fall detector (pose geometry + temporal state machine), fully causal.

Per frame it measures torso angle, box aspect, body height relative to the person's own upright height
and the body center. A state machine then requires

    STANDING / WALKING / SITTING -> FALLING (rapid descent) -> LYING held for `confirm_s` -> FALL_DETECTED

Rapid descent = within the last `descent_window_s`: body center moving down fast, OR height ratio lost
`drop_ratio`, OR torso rotated `drop_angle` degrees. Height loss matters for falls toward the camera,
where the center barely moves in the image. Reaching the floor without a rapid descent ends in LYING
with no alarm (lying down on purpose); getting back up during FALLING cancels it.
Timers run on wall-clock frames, so pose dropouts (frequent on lying people) do not freeze them.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

STATES = ["UNKNOWN", "STANDING", "WALKING", "SITTING", "FALLING", "LYING", "FALL_DETECTED"]
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class Params:
    fps: int = 30
    img_w: int = 640
    img_h: int = 480
    kp_conf: float = 0.3          # keypoint confidence to use a joint
    # posture
    upright_angle: float = 30.0   # torso angle (deg from vertical) below which the body is upright
    lying_angle: float = 55.0     # torso angle above which the body is horizontal
    lying_aspect: float = 1.2     # box width / height above which the body is horizontal
    low_ratio: float = 0.4        # box height / upright height below which the person is down (crouch ~0.45)
    bend_ratio: float = 0.55      # a horizontal torso only counts as down below this ratio (else bending over)
    sit_ratio: float = 0.8        # below this (but not low) and torso upright -> sitting
    walk_speed: float = 0.35      # horizontal center speed (heights/s) that counts as walking
    hold_s: float = 0.3           # posture must persist this long to change state
    # rapid descent (enter FALLING)
    descent_window_s: float = 0.6
    v_fall: float = 0.9           # downward center speed (upright heights per second over 0.3 s)
    drop_ratio: float = 0.4       # loss of height ratio within the window
    drop_angle: float = 40.0      # torso rotation within the window
    # event logic
    fall_window_s: float = 2.0    # max FALLING duration before giving up
    confirm_s: float = 0.5        # LYING after a fall must last this long to raise the alarm
    recover_s: float = 1.0        # upright this long to leave LYING / FALL_DETECTED
    ref_alpha: float = 0.05       # EMA rate of the upright reference height


def _mid(kp, a, b, conf):
    pts = [kp[i, :2] for i in (a, b) if kp[i, 2] >= conf]
    return np.mean(pts, axis=0) if pts else None


class FallStateMachine:
    def __init__(self, p=Params()):
        self.p = p
        self.state = "UNKNOWN"
        self.t = 0                    # frame clock
        self.t_enter = 0              # frame the current state started
        self.hist = deque()           # (t, cx, cy, angle, h) over the descent window
        self.h_ref = None
        self.counts = {"up": 0, "sit": 0, "down": 0}
        self.fell = False
        self.prev_upright_state = "STANDING"

    # ---------- per-frame measurements ----------
    def measure(self, kp, box):
        p = self.p
        has_box = not np.isnan(box[0])
        has_kp = not np.all(np.isnan(kp[:, 0]))
        sh = _mid(kp, L_SH, R_SH, p.kp_conf) if has_kp else None
        hip = _mid(kp, L_HIP, R_HIP, p.kp_conf) if has_kp else None
        if not has_box and hip is None:
            return None
        angle = None
        if sh is not None and hip is not None and np.linalg.norm(sh - hip) > 5:
            v = sh - hip
            angle = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))  # 0 upright, 90 horizontal
            angle = 180 - angle if angle > 90 else angle             # head-down counts as horizontal-ish
        x1, y1, x2, y2 = box[:4]
        w, h = x2 - x1, y2 - y1
        cx, cy = hip if hip is not None else ((x1 + x2) / 2, (y1 + y2) / 2)
        return dict(angle=angle, aspect=w / h if has_box and h > 0 else None, h=float(h) if has_box else None,
                    cx=float(cx), cy=float(cy))

    def posture(self, m):
        """Instantaneous posture from geometry: 'down', 'sit', 'up' or None (ambiguous)."""
        p = self.p
        ratio = m["h"] / self.h_ref if m["h"] and self.h_ref else None
        torso_flat = m["angle"] is not None and m["angle"] > p.lying_angle
        body_wide = m["aspect"] is not None and m["aspect"] > p.lying_aspect
        bending = torso_flat and ratio is not None and ratio >= p.bend_ratio
        if body_wide or (torso_flat and not bending) or (ratio is not None and ratio < p.low_ratio):
            return "down", ratio
        upright = m["angle"] is None or m["angle"] < p.upright_angle
        if upright and (ratio is None or ratio >= p.sit_ratio):
            return "up", ratio
        if ratio is not None and ratio < p.sit_ratio:
            return "sit", ratio
        return None, ratio

    def descent(self, m):
        """Signals of a rapid descent over the recent window (all causal)."""
        p = self.p
        out = dict(v_down=0.0, v_x=0.0, d_ratio=0.0, d_angle=0.0)
        if not self.hist or not self.h_ref:
            return out
        lag = round(0.3 * p.fps)
        recent = [e for e in self.hist if self.t - e[0] <= lag]
        if recent:
            t0, cx0, cy0, _, _ = recent[0]
            dt = max(self.t - t0, 1) / p.fps
            if dt >= 0.15:
                out["v_down"] = (m["cy"] - cy0) / dt / self.h_ref
                out["v_x"] = abs(m["cx"] - cx0) / dt / self.h_ref
        hs = [e[4] for e in self.hist if e[4]]
        if hs and m["h"]:
            out["d_ratio"] = (max(hs) - m["h"]) / self.h_ref
        angs = [e[3] for e in self.hist if e[3] is not None]
        if angs and m["angle"] is not None:
            out["d_angle"] = m["angle"] - min(angs)
        return out

    # ---------- state machine ----------
    def _go(self, s):
        self.state, self.t_enter = s, self.t

    def elapsed(self):
        return (self.t - self.t_enter) / self.p.fps

    def step(self, kp, box):
        """Feed one frame; returns (state, alarm_raised_this_frame, features dict or None)."""
        p = self.p
        self.t += 1
        while self.hist and self.t - self.hist[0][0] > p.descent_window_s * p.fps:
            self.hist.popleft()
        m = self.measure(kp, box)
        alarm = False
        if m is None:  # nothing seen: keep state, but time still passes
            if self.state == "LYING" and self.fell and self.elapsed() >= p.confirm_s:
                self._go("FALL_DETECTED")
                alarm = True
            return self.state, alarm, None

        if self.h_ref is None:  # start only once a fully visible upright person is seen
            x1, y1, x2, y2 = box[:4]
            inside = x1 > 5 and y1 > 5 and x2 < p.img_w - 5 and y2 < p.img_h - 5
            if m["h"] and inside and m["angle"] is not None and m["angle"] < p.upright_angle:
                self.h_ref = m["h"]
            else:
                return self.state, False, dict(m, ratio=None, posture=None, v_down=0.0, v_x=0.0,
                                               d_ratio=0.0, d_angle=0.0)
        post, ratio = self.posture(m)
        for k in self.counts:
            self.counts[k] = self.counts[k] + 1 if post == k else 0
        d = self.descent(m)
        self.hist.append((self.t, m["cx"], m["cy"], m["angle"], m["h"]))

        hold = round(p.hold_s * p.fps)
        fast = d["v_down"] > p.v_fall or d["d_ratio"] > p.drop_ratio or \
            (d["d_angle"] > p.drop_angle and (m["angle"] or 0) > p.upright_angle)
        s = self.state

        if s in ("UNKNOWN", "STANDING", "WALKING", "SITTING"):
            if fast and s != "UNKNOWN":
                self.prev_upright_state = s
                self._go("FALLING")
                if post == "down":          # already on the floor when the descent is recognised
                    self.counts["down"] = max(self.counts["down"], 1)
            elif self.counts["down"] >= hold:
                self.fell = False
                self._go("LYING")          # reached the floor slowly: lying down on purpose
            elif self.counts["sit"] >= hold:
                if s != "SITTING":
                    self._go("SITTING")
            elif self.counts["up"] >= hold or (s == "UNKNOWN" and post == "up"):
                new = "WALKING" if d["v_x"] > p.walk_speed else "STANDING"
                if s != new:
                    self._go(new)
        if self.state == "FALLING":
            if self.counts["down"] >= hold:
                self.fell = True
                self._go("LYING")
            elif self.counts["up"] >= hold and not fast and self.elapsed() > p.hold_s:
                self._go(self.prev_upright_state)   # recovered: cancel
            elif self.elapsed() > p.fall_window_s:
                self._go("SITTING" if post == "sit" else self.prev_upright_state)
        elif self.state == "LYING":
            if self.fell and self.elapsed() >= p.confirm_s:
                self._go("FALL_DETECTED")
                alarm = True
            elif self.counts["up"] >= p.recover_s * p.fps:
                self._go("STANDING")
            elif self.counts["sit"] >= p.recover_s * p.fps:
                self._go("SITTING")
        elif self.state == "FALL_DETECTED":
            if self.counts["up"] >= p.recover_s * p.fps:
                self._go("STANDING")

        # update the upright reference height only while clearly standing
        if self.state in ("STANDING", "WALKING") and post == "up" and m["h"]:
            self.h_ref += p.ref_alpha * (m["h"] - self.h_ref)

        return self.state, alarm, dict(m, ratio=ratio, posture=post, **d)


def run(kp_seq, box_seq, p=Params()):
    """Run on a whole cached sequence. Returns per-frame states, alarm frame numbers (1-based), features."""
    sm = FallStateMachine(p)
    states, alarms, feats = [], [], []
    for i, (kp, box) in enumerate(zip(kp_seq, box_seq)):
        s, a, f = sm.step(kp, box)
        states.append(s)
        feats.append(f)
        if a:
            alarms.append(i + 1)
    return states, alarms, feats
