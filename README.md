# UR-Fall-Detection

Human fall detection from video with graph machine learning: MediaPipe skeletons as spatio-temporal graphs, classified with graph kernels, graph embeddings and GNNs.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Download the data

```bash
python scripts/download_urfd.py            # everything (~4.4 GB)
python scripts/download_urfd.py --no-rgb   # skip the RGB frames (~100 MB)
```

This fetches the [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) (camera 0 only) into `data/raw/urfd/`:

| Folder | Content |
|---|---|
| `rgb/<seq>/` | 640x480 PNG frames, 30 fps (the input for pose estimation) |
| `videos/` | preview mp4s (depth and RGB side by side, 640x240) |
| `sync/` | per-frame timestamps and accelerometer magnitude |
| `labels/` | per-frame labels: -1 not lying, 0 transition, 1 lying on the floor (+ 8 depth features) |

70 sequences: `fall-01` to `fall-30` and `adl-01` to `adl-40` (daily activities, some include lying down on purpose).
Camera 1 (ceiling mounted) is skipped since 2D pose estimation works poorly from above.
The script is safe to re-run: finished sequences are skipped.

`data/` is git-ignored, so each of us downloads it locally.

## Pose model

MediaPipe 1.x uses the Tasks API and needs the model file:

```bash
mkdir -p models
curl -L -o models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Dataset citation

B. Kwolek, M. Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014. License CC BY-NC-SA 4.0 (non-commercial academic use).

## B0: rule-based baseline (pose + state machine)

Interpretable reference baseline, no training. A per-person state machine (`src/fallcv/rule.py`) needs
`STANDING/WALKING/SITTING -> FALLING (rapid descent) -> LYING held 0.5 s -> FALL_DETECTED`.
Lying down slowly gives `LYING` with no alarm; getting back up during `FALLING` cancels it.

```bash
python scripts/extract_poses.py --extractor mediapipe   # or yolo (needs torch + ultralytics), cached in data/interim/poses/
python scripts/run_b0.py --extractor mediapipe --sweep  # metrics -> results/
python scripts/render_b0.py fall-03                     # overlay video -> results/videos/
```

Event metrics: a fall is hit if the first alarm is within 3 s of its onset (first transition frame).
Any alarm in an ADL is a false alarm. Thresholds were set by hand while looking at all 70 clips,
so these numbers are optimistic (no held-out split yet).

| Pose extractor | Falls hit | ADLs with alarm | FA / hour | Median latency | Lying-frame F1 |
|---|---|---|---|---|---|
| MediaPipe (VIDEO mode) | 17/30 | 10/40 | 121 | 1.4 s | 0.86 |
| YOLO11m-pose | 13/30 | 11/40 | 133 | 1.6 s | 0.83 |

Notes: URFD fall clips end about 0.7 s (median) after the person lands, so a long "lying" confirmation
cannot be observed; with 1 s confirmation recall drops to 12/30. Most remaining false alarms are ADLs where
the actor lies down on purpose in about 1 s, which geometry alone cannot separate from a fall.
