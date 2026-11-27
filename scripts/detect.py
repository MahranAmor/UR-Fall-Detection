"""Fall detector: the skeleton LSTM of notebooks 12, 13b and 14.

    python scripts/detect.py my_video.mp4         one video: says FALL or NO FALL, saves a video with the result and a plot in results/demo/
    python scripts/detect.py data/raw/home        every video of a folder, and a summary table (results/demo/summary.csv)
    python scripts/detect.py some/folder/of/pngs --fps 20    a folder of images (one video)
    python scripts/detect.py --camera 0           live, from the webcam (q = quit, r = reset the alarm)

How it works (the same steps as in the notebooks):
  1. MediaPipe finds 13 joints of the person in every image;
  2. we keep 10 images per second, and the last 1.5 s (15 images) of skeleton make a "window";
  3. 3 LSTMs give a fall score for each window, we take their mean;
  4. the score is smoothed (median of the last 3) and an ALARM starts when it goes above the threshold.
The model and the threshold come from notebook 14 (models/fall_lstm.pt).
"""
import os
os.environ.setdefault("GLOG_minloglevel", "2"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # fewer MediaPipe messages
from pathlib import Path
from collections import deque
import argparse, re, time
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
POSE_MODEL = ROOT / "models" / "pose_landmarker_full.task"
FALL_MODEL = ROOT / "models" / "fall_lstm.pt"
OUT = ROOT / "results" / "demo"
JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]        # nose, shoulders, elbows, wrists, hips, knees, ankles (MediaPipe numbers)
BONES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]
T, STEP = 15, 0.1                                                     # 15 images in a window, one image every 0.1 s
VIDEO_TYPES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_SIDE = 1280                                                       # bigger videos (phones) are made smaller first: faster, and like our datasets


# ---------- 1. skeletons ----------
def pose_detector():
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    global _mp; _mp = mp
    options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(POSE_MODEL)), running_mode=vision.RunningMode.VIDEO, num_poses=1)
    return vision.PoseLandmarker.create_from_options(options)

def skeleton_of(detector, image, ms):
    """13 joints (x, y between 0 and 1) of one BGR image, or NaN if nobody is found. ms = time of the image, in milliseconds."""
    result = detector.detect_for_video(_mp.Image(image_format=_mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), int(ms))
    if not result.pose_landmarks: return np.full((13, 2), np.nan, np.float32)
    return np.array([[result.pose_landmarks[0][k].x, result.pose_landmarks[0][k].y] for k in JOINTS], np.float32)

def shrink(image):
    h, w = image.shape[:2]; s = MAX_SIDE / max(h, w)
    return image if s >= 1 else cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


# ---------- 2. windows (exactly as in notebook 13b) ----------
def grid_of(n, fps):
    """10 images per second: the nearest image of t = 0, 0.1, 0.2 ... seconds (notebook 13a)."""
    return np.unique(np.round(np.arange(0, (n - 1) / fps + 1e-6, STEP) * fps).astype(int))

def aspect_of(width, height):
    """x is multiplied by this to be in image heights, like y (notebooks 13a and 13b)."""
    return round(width * 480 / height) / 480

def window_at(P, ok, t, k):
    """The window that ends at kept image k. P = skeletons of the kept images (x in image heights, last known skeleton kept),
    ok = was the person found in that image, t = times in seconds. Gives 15 images x 13 joints x (x, y, change of x, change of y, found)."""
    pos = np.arange(k - T + 1, k + 1); w = np.clip(pos, 0, None); p = P[w]
    if np.isnan(p[:, 0, 0]).all(): return np.zeros((T, 13, 5), np.float32)                 # nobody seen yet
    hips, shoulders = p[:, 7:9].mean(1), p[:, 1:3].mean(1)
    q = (p - np.nanmean(hips, axis=0)) / max(np.nanpercentile(np.linalg.norm(shoulders - hips, axis=1), 90), 0.05)   # centred on the hips, in body sizes
    d = np.zeros_like(q); d[1:] = (q[1:] - q[:-1]) / np.maximum(np.diff(t[w]), 1e-3)[:, None, None] / 3                 # change per second
    f = np.broadcast_to((ok[w] & (pos >= 0))[:, None, None], (T, 13, 1))
    return np.nan_to_num(np.concatenate([q, d, f], -1)).clip(-10, 10).astype(np.float32)

def windows_of(P, index, fps, aspect):
    """All the windows of a video. P = skeletons of ALL its images (NaN = nobody), index = the kept images (10 per second)."""
    P = np.asarray(P, float).copy(); P[..., 0] *= aspect
    ok = ~np.isnan(P[:, 0, 0])
    P = pd.DataFrame(P.reshape(len(P), -1)).ffill().to_numpy().reshape(P.shape)          # keep the last known skeleton
    P, ok, t = P[index], ok[index], np.asarray(index) / fps
    return np.stack([window_at(P, ok, t, k) for k in range(len(index))]) if len(index) else np.zeros((0, T, 13, 5), np.float32)


# ---------- 3. the model ----------
class LSTM(nn.Module):                                                                    # notebook 12
    def __init__(self, n=2):
        super().__init__(); self.lstm = nn.LSTM(13 * 4 + 1, 64, batch_first=True); self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n))
    def forward(self, w): return self.out(self.lstm(torch.cat([w[..., :4].flatten(2), w[:, :, 0, 4:5]], 2))[0][:, -1])

class FallModel:
    """The 3 LSTMs and the threshold saved by notebook 14."""
    def __init__(self, path=FALL_MODEL, device="cpu"):
        if not Path(path).exists(): raise SystemExit(f"{path} not found: run notebook 14 first.")
        saved = torch.load(path, map_location=device, weights_only=True)
        self.nets = []
        for state in saved["nets"]:
            net = LSTM(2); net.load_state_dict(state); self.nets.append(net.to(device).eval())
        self.threshold, self.device, self.info = float(saved["threshold"]), device, saved
    def scores(self, windows):
        """Fall score (0 to 1) of each window: the mean of the 3 LSTMs."""
        if len(windows) == 0: return np.zeros(0)
        with torch.no_grad():
            w = torch.from_numpy(np.ascontiguousarray(windows, np.float32)).to(self.device)
            return np.mean([torch.cat([net(b).softmax(1)[:, 1] for b in torch.split(w, 512)]).cpu().numpy() for net in self.nets], 0)

def smooth(score): return pd.Series(score).rolling(3, min_periods=1).median().to_numpy()   # median of the last 3 kept images (0.3 s)

def alarm_times(s, t, threshold):
    on = s > threshold
    return t[on & ~np.r_[False, on[:-1]]]                                                # the moments when the score goes above the threshold


# ---------- 4. one video ----------
def images_of(path):
    """The images of a video file, or of a folder of images, in order (BGR)."""
    path = Path(path)
    if path.is_dir():
        natural = lambda p: [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]   # image2 before image10
        for p in sorted((p for p in path.iterdir() if p.suffix.lower() in IMAGE_TYPES), key=natural): yield cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(path))
        while True:
            ok, image = cap.read()
            if not ok: break
            yield image
        cap.release()

def fps_of(path, fps=None):
    if fps: return float(fps)
    if Path(path).is_dir(): return 30.0
    f = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS)
    return float(f) if f and 1 <= f <= 240 else 30.0

def analyse(path, model, fps=None):
    """Skeletons, scores and alarms of one video."""
    fps = fps_of(path, fps); P, size = [], None
    with pose_detector() as detector:
        for i, image in enumerate(images_of(path)):
            image = shrink(image); size = size or (image.shape[1], image.shape[0])
            P.append(skeleton_of(detector, image, i * 1000 / fps))
    if not P: raise SystemExit(f"cannot read any image from {path}")
    P = np.array(P); index = grid_of(len(P), fps); t = index / fps
    score = model.scores(windows_of(P, index, fps, aspect_of(*size))); s = smooth(score)
    return {"path": Path(path), "fps": fps, "size": size, "skeletons": P, "index": index, "t": t, "score": score, "smooth": s,
            "alarms": alarm_times(s, t, model.threshold), "threshold": model.threshold}

def sentence(r):
    found = np.mean(~np.isnan(r["skeletons"][:, 0, 0]))
    if len(r["alarms"]): return f"FALL detected at {r['alarms'][0]:.1f} s  (score {r['smooth'].max():.2f}, threshold {r['threshold']:.2f}, person found in {found:.0%} of the images)"
    return f"NO FALL  (highest score {r['smooth'].max():.2f}, threshold {r['threshold']:.2f}, person found in {found:.0%} of the images)"


# ---------- 5. pictures ----------
def draw(image, skeleton, score, threshold, alarm_at, now, history):
    """The result around one image: the answer ABOVE it, the skeleton on it, the last 10 s of score BELOW it (nothing hides the person)."""
    h, w = image.shape[:2]; u = w / 640                                                    # u = size unit
    top, bottom = round(40 * u), round(70 * u)
    out = np.zeros((h + top + bottom, w, 3), np.uint8); out[top:top + h] = image
    font, thick, red = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(2 * u)), (0, 0, 255)
    if not np.isnan(skeleton[0, 0]):
        q = (skeleton * [w, h]).astype(int) + [0, top]
        for a, b in BONES: cv2.line(out, tuple(q[a]), tuple(q[b]), (0, 255, 0), thick, cv2.LINE_AA)
        for p in q: cv2.circle(out, tuple(p), max(2, round(3 * u)), (0, 200, 255), -1, cv2.LINE_AA)
    fall = alarm_at is not None
    cv2.putText(out, f"FALL DETECTED at {alarm_at:.1f} s" if fall else "no fall", (round(10 * u), round(28 * u)), font, 0.8 * u, red if fall else (0, 190, 0), thick, cv2.LINE_AA)
    label = f"score {score:.2f}"; (tw, _), _ = cv2.getTextSize(label, font, 0.6 * u, max(1, round(u)))
    cv2.putText(out, label, (w - tw - round(10 * u), round(27 * u)), font, 0.6 * u, red if score > threshold else (255, 255, 255), max(1, round(u)), cv2.LINE_AA)
    y0, y1 = top + h + round(8 * u), top + h + bottom - round(6 * u)                        # the curve of the last 10 s
    level = lambda s: round(y1 - (y1 - y0) * min(max(s, 0), 1))
    cv2.line(out, (0, level(threshold)), (w, level(threshold)), red, 1)
    cv2.putText(out, "fall score, last 10 s (red line = threshold)", (round(6 * u), y1 - round(4 * u)), font, 0.4 * u, (170, 170, 170), 1, cv2.LINE_AA)
    pts = np.array([(round(w * (1 - (now - tt) / 10)), level(s)) for tt, s in history if now - tt <= 10], np.int32)
    if len(pts) > 1: cv2.polylines(out, [pts], False, (0, 200, 255), thick, cv2.LINE_AA)
    if fall: cv2.rectangle(out, (0, top), (w - 1, top + h - 1), red, max(3, round(5 * u)))
    return out

def save_video(r, file):
    """The video with the result drawn on it (mp4; it opens in VLC and in most players)."""
    writer, first = None, r["alarms"][0] if len(r["alarms"]) else np.inf
    for i, image in enumerate(images_of(r["path"])):
        if i >= len(r["skeletons"]): break
        now = i / r["fps"]; k = max(np.searchsorted(r["index"], i, side="right") - 1, 0)
        history = list(zip(r["t"][:k + 1], r["smooth"][:k + 1]))
        frame = draw(shrink(image), r["skeletons"][i], r["smooth"][k], r["threshold"], first if now >= first else None, now, history)
        writer = writer or cv2.VideoWriter(str(file), cv2.VideoWriter_fourcc(*"mp4v"), r["fps"], (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    if writer: writer.release()

def save_plot(r, file):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(r["t"], r["score"], color="lightgrey", label="score of each image")
    ax.plot(r["t"], r["smooth"], color="tab:blue", lw=2, label="smoothed score")
    ax.axhline(r["threshold"], color="red", ls="--", label=f"threshold {r['threshold']:.2f}")
    for a in r["alarms"][:1]: ax.axvline(a, color="red", lw=2); ax.text(a, 0.5, f" ALARM {a:.1f} s", color="red")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("time (s)"); ax.set_ylabel("fall score"); ax.set_title(r["path"].name); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(file, dpi=110); plt.close(fig)


# ---------- 6. live, from a camera ----------
def live(camera, model, save=None):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened(): raise SystemExit(f"cannot open camera {camera}")
    P, OK, TT, last_scores, history = deque(maxlen=T), deque(maxlen=T), deque(maxlen=T), deque(maxlen=3), deque(maxlen=200)
    last, tick, s_now, on_before, alarm_at, last_ms, writer, aspect = np.full((13, 2), np.nan), 0.0, 0.0, False, None, -1, None, None
    with pose_detector() as detector:
        start, n = time.perf_counter(), 0; print("press q to quit, r to reset the alarm")
        while True:
            ok, image = cap.read()
            if not ok: break
            n += 1
            image = shrink(image); now = time.perf_counter() - start; ms = max(int(now * 1000), last_ms + 1); last_ms = ms
            aspect = aspect or aspect_of(image.shape[1], image.shape[0])
            sk = skeleton_of(detector, image, ms); found = not np.isnan(sk[0, 0])
            if found: last = sk
            if now >= tick:                                                               # one kept image every 0.1 s
                while tick <= now: tick += STEP
                q = last.astype(float).copy(); q[:, 0] *= aspect
                P.append(q); OK.append(found); TT.append(now)
                last_scores.append(model.scores(window_at(np.array(P), np.array(OK), np.array(TT), len(P) - 1)[None])[0])
                s_now = float(np.median(last_scores)); history.append((now, s_now))
                on = s_now > model.threshold
                if on and not on_before and alarm_at is None: alarm_at = now; print(f"FALL DETECTED at {now:.1f} s (score {s_now:.2f})")
                on_before = on
            frame = draw(image, sk, s_now, model.threshold, alarm_at, now, history)
            if save:
                writer = writer or cv2.VideoWriter(str(save), cv2.VideoWriter_fourcc(*"mp4v"), 15, (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            cv2.imshow("fall detector (q = quit, r = reset)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("r"): alarm_at = None
    cap.release(); cv2.destroyAllWindows()
    if writer: writer.release()
    took = time.perf_counter() - start; print(f"{n} images in {took:.0f} s: {n / max(took, 1e-9):.0f} images per second")


# ---------- 7. command line ----------
def main():
    ap = argparse.ArgumentParser(description="Fall / no fall for a video, a folder of videos, or a camera.")
    ap.add_argument("path", nargs="?", help="a video, a folder of videos, or a folder of images")
    ap.add_argument("--camera", type=int, help="number of the camera for the live mode (0 = the laptop camera)")
    ap.add_argument("--fps", type=float, help="images per second (only needed for a folder of images; default 30)")
    ap.add_argument("--out", default=str(OUT), help="where to save the results (default results/demo)")
    ap.add_argument("--no-video", action="store_true", help="do not save the video with the result (faster)")
    ap.add_argument("--save", help="live mode: also record the result in this mp4 file")
    a = ap.parse_args()
    model = FallModel()
    if a.camera is not None: return live(a.camera, model, a.save)
    if not a.path: ap.error("give a video, a folder, or --camera 0")
    path = Path(a.path); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if path.is_dir() and not any(p.suffix.lower() in IMAGE_TYPES for p in path.iterdir()):
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_TYPES)
        if not files: raise SystemExit(f"no video in {path}")
    else: files = [path]
    rows = []
    for f in files:
        t0 = time.perf_counter(); r = analyse(f, model, a.fps); name = f.stem if f.is_file() else f.name
        print(f"{f.name}: {sentence(r)}")
        save_plot(r, out / f"{name}_score.png")
        if not a.no_video: save_video(r, out / f"{name}_result.mp4")
        rows.append({"video": str(f), "result": "FALL" if len(r["alarms"]) else "no fall", "first alarm (s)": round(r["alarms"][0], 2) if len(r["alarms"]) else None,
                     "highest score": round(float(r["smooth"].max()), 3), "threshold": round(r["threshold"], 3),
                     "person found": round(float(np.mean(~np.isnan(r["skeletons"][:, 0, 0]))), 2), "seconds": round(len(r["skeletons"]) / r["fps"], 1),
                     "time to analyse (s)": round(time.perf_counter() - t0, 1)})
    if len(rows) > 1:
        table = pd.DataFrame(rows); table.to_csv(out / "summary.csv", index=False)
        print(f"\n{(table.result == 'FALL').sum()} of {len(table)} videos with a FALL. Table: {out / 'summary.csv'}")
    print(f"pictures and videos in {out}")

if __name__ == "__main__":
    main()
