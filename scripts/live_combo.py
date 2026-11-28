"""Fall detector for live use: the skeleton LSTM (motion) + a 3D posture check (confirmation).

    python scripts/live_combo.py --camera 0                 live (q = quit, r = reset)
    python scripts/live_combo.py --video clip.mp4           a video file (same logic, image times from the file)
    python scripts/live_combo.py --video clip.mp4 --log out.csv --save out.mp4 --no-show

Why: the LSTM alone fires on skeleton glitches (e.g. MediaPipe jumping to a vase), and posture rules alone
confuse "far from the camera" with "low". Here
  1. MediaPipe looks for 2 people and we keep the real one (visible body, no jump across the image);
  2. the posture comes from MediaPipe's 3D skeleton in metres (torso angle, knee angle, hip height above
     the feet), so it does not change with the distance to the camera;
  3. the LSTM of notebook 14 (models/fall_lstm.pt) gives a fall score every 0.1 s on the cleaned skeleton;
  4. a high score only starts a CHECK; the ALARM needs the person to be down (lying, sitting on the floor;
     on all fours only if the LSTM was very sure) for 1 s
     within 4 s, or to vanish low in the image (fell out of view). Standing up again cancels it.
"""
import os
os.environ.setdefault("GLOG_minloglevel", "2"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse, csv, sys, time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect  # noqa: E402  (the LSTM, its windows and the model file)

CORE = [11, 12, 23, 24, 25, 26]                    # shoulders, hips, knees (MediaPipe numbers)
DOWN = {"LYING", "SITTING ON FLOOR"}     # ON FLOOR (all fours) only counts after a very sure LSTM score
COLORS = {"STANDING": (80, 200, 80), "SITTING": (200, 160, 60), "SITTING ON FLOOR": (0, 200, 255), "BENDING": (200, 200, 80),
          "ON FLOOR": (0, 165, 255), "LYING": (0, 220, 255), "?": (150, 150, 150), "NOBODY": (150, 150, 150)}
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


class Settings:
    trigger = 0.85          # LSTM score that starts a check (the LSTM's own threshold 0.989 misses falls at the image edge)
    sure = 0.989            # above this (the threshold of notebook 14), being on all fours also confirms
    check_s = 4.0           # how long after the trigger we wait for the person to be down
    down_s = 1.0            # down this long during a check -> ALARM (short dataset clips may end before this)
    lost_s = 0.8            # vanished this long during a check, last seen low in the image -> ALARM
    low_y = 0.65            # "low in the image" (hips below this fraction of the height)
    recover_s = 1.5         # standing this long ends an alarm or a check
    max_jump = 0.25         # a new skeleton farther than this (image widths) from the last one is ignored ...
    jump_grace_s = 0.7      # ... unless the person was lost for this long
    smooth_s = 0.5          # posture = most common label of the last 0.5 s


def angle(a, b):
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def posture_features(img, world):
    """img: 33 x (x, y, visibility) in 0-1 image units; world: 33 x (x, y, z) in metres (y down, centred on the hips)."""
    up = np.array([0, -1.0, 0])
    sh, hip = world[[11, 12]].mean(0), world[[23, 24]].mean(0)
    knee = np.mean([angle(world[23] - world[25], world[27] - world[25]), angle(world[24] - world[26], world[28] - world[26])])
    return dict(torso=angle(sh - hip, up), knee=float(knee),
                hip_h=float(world[[27, 28, 29, 30, 31, 32], 1].max() - hip[1]),   # hip height above the lowest foot point
                hip_y=float(img[[23, 24], 1].mean()), legs_vis=float(img[[25, 26], 2].max()))


def posture_of(f):
    """Thresholds measured on our own recording (demo1): standing hip 0.75-0.9 m, sofa 0.42-0.57 m, floor < 0.25 m."""
    if f is None: return "NOBODY"
    if f["torso"] >= 60:                                   # torso horizontal
        if f["hip_h"] < 0.25: return "LYING"
        if f["hip_h"] < 0.5: return "ON FLOOR"            # on all fours, crawling
        return "BENDING"                                   # legs still under the hips
    if f["torso"] < 40:                                    # torso upright
        if f["legs_vis"] < 0.3: return "?"                 # legs not visible: cannot tell standing from sitting
        if f["hip_h"] < 0.3: return "SITTING ON FLOOR"
        if f["knee"] > 145 and f["hip_h"] > 0.7: return "STANDING"
        return "SITTING"
    return "?"                                             # 40-60 degrees: in between, keep the previous label


class Tracker:
    """MediaPipe with 2 candidates: keep the visible body that continues the previous one."""
    def __init__(self, s):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp, self.s = mp, s
        self.det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(detect.POSE_MODEL)), running_mode=vision.RunningMode.VIDEO, num_poses=2))
        self.last_xy, self.last_t, self.last_ms = None, -1e9, -1

    def __call__(self, bgr, t):
        ms = max(int(t * 1000), self.last_ms + 1); self.last_ms = ms
        r = self.det.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), ms)
        best, best_key = None, None
        for lm, wl in zip(r.pose_landmarks, r.pose_world_landmarks):
            img = np.array([[p.x, p.y, p.visibility] for p in lm]); vis = img[CORE, 2].mean()
            if vis < 0.5: continue
            xy = img[[23, 24], :2].mean(0)
            jump = 0 if self.last_xy is None else np.linalg.norm(xy - self.last_xy)
            if jump > self.s.max_jump and t - self.last_t < self.s.jump_grace_s: continue
            key = (round(jump, 2), -vis)
            if best_key is None or key < best_key:
                best, best_key = (img, np.array([[p.x, p.y, p.z] for p in wl])), key
        if best is not None:
            self.last_xy, self.last_t = best[0][[23, 24], :2].mean(0), t
        return best


class FallDetector:
    def __init__(self, s=Settings()):
        self.s, self.model, self.tracker = s, detect.FallModel(), Tracker(s)
        self.reset()

    def reset(self):
        self.P, self.OK, self.TT = deque(maxlen=detect.T), deque(maxlen=detect.T), deque(maxlen=detect.T)
        self.scores, self.labels = deque(maxlen=3), deque()
        self.last13, self.tick, self.score = np.full((13, 2), np.nan), 0.0, 0.0
        self.state, self.t_state, self.posture, self.peak = "NORMAL", 0.0, "NOBODY", 0.0
        self.since = {"down": None, "up": None, "lost": None}
        self.last_hip_y, self.alarms, self.aspect = None, [], None

    def _hold(self, key, active, t):
        if not active: self.since[key] = None; return 0.0
        if self.since[key] is None: self.since[key] = t
        return t - self.since[key]

    def step(self, bgr, t):
        self.aspect = self.aspect or detect.aspect_of(bgr.shape[1], bgr.shape[0])
        person = self.tracker(bgr, t)
        f = posture_features(*person) if person is not None else None
        if f is not None: self.last_hip_y = f["hip_y"]
        # posture: most common label of the last smooth_s seconds ("?" keeps the previous one)
        raw = posture_of(f)
        self.labels.append((t, raw))
        while self.labels and t - self.labels[0][0] > self.s.smooth_s: self.labels.popleft()
        votes = Counter(l for _, l in self.labels if l != "?")
        if votes: self.posture = votes.most_common(1)[0][0]
        # LSTM, 10 times per second, on the cleaned skeleton (exactly the windows of detect.py)
        if person is not None: self.last13 = person[0][detect.JOINTS, :2]
        if t >= self.tick:
            while self.tick <= t: self.tick += detect.STEP
            q = self.last13.astype(float).copy(); q[:, 0] *= self.aspect
            self.P.append(q); self.OK.append(person is not None); self.TT.append(t)
            w = detect.window_at(np.array(self.P), np.array(self.OK), np.array(self.TT), len(self.P) - 1)
            self.scores.append(self.model.scores(w[None])[0]); self.score = float(np.median(self.scores))
        # decision
        if self.state == "CHECK": self.peak = max(self.peak, self.score)
        down = self._hold("down", self.posture in DOWN or (self.posture == "ON FLOOR" and self.peak >= self.s.sure), t)
        up = self._hold("up", self.posture == "STANDING", t)
        lost = self._hold("lost", person is None, t)
        alarm = False
        if self.state == "NORMAL" and self.score > self.s.trigger:
            self.state, self.t_state, self.peak = "CHECK", t, self.score
        elif self.state == "CHECK":
            fell_out = lost >= self.s.lost_s and (self.last_hip_y or 0) > self.s.low_y
            if down >= self.s.down_s or fell_out:
                self.state, self.t_state, alarm = "ALARM", t, True; self.alarms.append(t)
            elif t - self.t_state > self.s.check_s or (up >= self.s.recover_s and t - self.t_state > 1.0):
                self.state, self.t_state = "NORMAL", t
        elif self.state == "ALARM" and up >= self.s.recover_s:
            self.state, self.t_state = "NORMAL", t
        return dict(t=t, posture=self.posture, raw=raw, score=self.score, state=self.state, alarm=alarm, person=person, f=f)


def draw(img, r, n_alarms):
    h, w = img.shape[:2]
    if r["person"] is not None:
        p = r["person"][0]
        for a, b in BONES:
            if p[a, 2] > 0.3 and p[b, 2] > 0.3:
                cv2.line(img, (int(p[a, 0] * w), int(p[a, 1] * h)), (int(p[b, 0] * w), int(p[b, 1] * h)), (255, 255, 255), 2)
    alarm = r["state"] == "ALARM"
    cv2.rectangle(img, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.putText(img, r["posture"], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS.get(r["posture"], (255, 255, 255)), 2)
    st_col = (40, 40, 230) if alarm else (0, 165, 255) if r["state"] == "CHECK" else (180, 180, 180)
    cv2.putText(img, r["state"], (w - 150, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, st_col, 2)
    f = r["f"]
    info = f"score {r['score']:.2f}  alarms {n_alarms}" + (f"  torso {f['torso']:.0f}  knee {f['knee']:.0f}  hip {f['hip_h']:.2f} m" if f else "")
    cv2.putText(img, info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    if alarm:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (40, 40, 230), 10)
        cv2.putText(img, "FALL!", (w // 2 - 110, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (40, 40, 230), 6)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int)
    ap.add_argument("--video")
    ap.add_argument("--save", help="record the annotated result to this mp4")
    ap.add_argument("--log", help="write one csv line per image")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    if a.video is None and a.camera is None: a.camera = 0
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened(): sys.exit("cannot open the camera / video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fd, writer, i, start = FallDetector(), None, 0, time.perf_counter()
    log = csv.writer(open(a.log, "w", newline="")) if a.log else None
    if log: log.writerow(["t", "posture", "raw", "score", "state", "torso", "knee", "hip_h", "hip_y"])
    print("q = quit, r = reset")
    while True:
        ok, img = cap.read()
        if not ok: break
        img = detect.shrink(img)
        t = i / fps if a.video else time.perf_counter() - start
        i += 1
        r = fd.step(img, t)
        if r["alarm"]: print(f"FALL DETECTED at {t:.1f} s (score {r['score']:.2f}, posture {r['posture']})")
        if log:
            f = r["f"] or {}
            log.writerow([f"{t:.2f}", r["posture"], r["raw"], f"{r['score']:.3f}", r["state"]] + [f"{f[k]:.2f}" if k in f else "" for k in ("torso", "knee", "hip_h", "hip_y")])
        frame = draw(img, r, len(fd.alarms))
        if a.save:
            writer = writer or cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps if a.video else 15, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if not a.no_show:
            cv2.imshow("fall detector: LSTM + posture (q quit, r reset)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord("r"): fd.reset()
    cap.release(); cv2.destroyAllWindows()
    if writer: writer.release()
    print(f"{i} images, alarms at {[round(x, 1) for x in fd.alarms]} s")


if __name__ == "__main__":
    main()
