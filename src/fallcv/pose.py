"""Pose extractors with a common output: COCO-17 keypoints (x, y, conf in pixels) + person box (x1, y1, x2, y2, conf).

Missing detections are NaN. Results are cached per sequence in data/interim/poses/<extractor>/<seq>.npz.
"""
import cv2
import numpy as np

from .urfd import FPS, ROOT

CACHE = ROOT / "data" / "interim" / "poses"
COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
        "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class MediaPipePose:
    """MediaPipe Pose Landmarker in VIDEO mode (uses temporal tracking, causal)."""
    name = "mediapipe"

    def __init__(self, model=ROOT / "models" / "pose_landmarker_full.task", min_conf=0.3):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO,
            num_poses=1, min_pose_detection_confidence=min_conf, min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf)
        self.det = vision.PoseLandmarker.create_from_options(opts)
        self.t = 0

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.t += 1000 // FPS
        res = self.det.detect_for_video(img, self.t)
        if not res.pose_landmarks:
            return np.full((17, 3), np.nan), np.full(5, np.nan)
        lm = np.array([[p.x * w, p.y * h, p.visibility] for p in res.pose_landmarks[0]])
        vis = lm[lm[:, 2] > 0.3]
        pts = vis[:, :2] if len(vis) >= 4 else lm[:, :2]
        box = np.r_[pts.min(0), pts.max(0), lm[:, 2].mean()]
        return lm[MP_TO_COCO], box


class YoloPose:
    """Ultralytics YOLO-pose; keeps the most confident person (URFD has one actor per clip)."""
    name = "yolo"

    def __init__(self, weights=str(ROOT / "models" / "yolo11m-pose.pt"), conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, classes=[0], verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.full((17, 3), np.nan), np.full(5, np.nan)
        i = int(r.boxes.conf.argmax())
        box = np.r_[r.boxes.xyxy[i].cpu().numpy(), float(r.boxes.conf[i])]
        kp = r.keypoints.data[i].cpu().numpy()  # (17, 3)
        return kp, box


EXTRACTORS = {"mediapipe": MediaPipePose, "yolo": YoloPose}


def cache_path(extractor, seq):
    return CACHE / extractor / f"{seq}.npz"


def load(extractor, seq):
    d = np.load(cache_path(extractor, seq))
    return d["kp"], d["box"]
