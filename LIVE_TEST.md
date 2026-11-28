# Live test of the fall detector (instructions for Claude Code)

Goal: on this machine, set up the project and run the B0 fall detector live on the webcam, so the user can
try falls in front of the camera. No dataset and no GPU needed. Follow the steps in order and report the
result of each check to the user.

## 1. Check Python

Needs Python 3.12 (mediapipe 1.2.2 has no wheels for newer versions).

- Windows: `py -3.12 --version`
- Linux/macOS: `python3.12 --version`

If it is missing, ask the user to install Python 3.12 (python.org or `winget install Python.Python.3.12`), do not continue with another version.

## 2. Create the environment (from the repo root)

Windows:
```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install opencv-python mediapipe==1.2.2 numpy
```
Linux/macOS:
```
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install opencv-python mediapipe==1.2.2 numpy
```
Below, `PY` means `.venv\Scripts\python` (Windows) or `.venv/bin/python` (Linux/macOS).

If pip fails with an SSL "certificate verify failed" error, retry with `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

## 3. Download the pose model (~9 MB)

```
mkdir models
curl -L -o models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```
Check that `models/pose_landmarker_full.task` exists and is larger than 1 MB.

## 4. Smoke test (no webcam)

```
PY -c "import sys; sys.path.insert(0, 'src'); from fallcv import pose, rule; import numpy as np; d = pose.MediaPipePose(); print(d(np.zeros((480, 640, 3), np.uint8))[1]); print('OK')"
```
Expected: a line of `nan` values (no person in a black image), then `OK`. Ignore MediaPipe `W0000` / `INFO` log lines.

## 5. Find the webcam

```
PY -c "import cv2; print([i for i in range(4) if cv2.VideoCapture(i).read()[0]])"
```
Prints the working camera indices, e.g. `[0]`. If the list is empty: on Windows check Settings > Privacy > Camera
("Let desktop apps access your camera"); on macOS grant camera access to the terminal app; close other apps using the camera (Teams, Zoom, OBS).

## 6. Run live

```
PY scripts/live_demo.py --cam <index from step 5>
```
Run it in the background (it runs until the user presses `q` in its window) and watch its output.

- First line: `running at N fps (timers use this)`. Below 10 fps the detector gets unreliable; tell the user.
- Each detected fall prints `FALL DETECTED (hh:mm:ss)`.

Tell the user what to do (Claude cannot see the window or the room):

1. Place the camera 2 to 3 m away at chest height, whole body visible, good light.
2. Stand still, fully in view (head to feet), until the top-left label turns from `UNKNOWN` to `STANDING`.
3. Fall onto a mattress or soft surface and stay down about 1 s: the label goes `FALLING` -> `LYING` -> `FALL_DETECTED` with a red frame.
4. Also try things that should NOT alarm: sit on a chair, crouch, lie down slowly.
5. `r` resets (stand fully in view again), `q` quits.

Afterwards, summarise for the user: fps, how many falls they did vs how many `FALL DETECTED` lines appeared,
and any false alarms they report.

## Known limits (say these if results look off)

- B0 is a hand-tuned rule baseline: on the URFD dataset it caught 17/30 falls, and 10/40 daily activities raised an alarm.
- Lying down quickly looks like a fall to it. Falls straight toward the camera are harder.
- It tracks one person. Keep other people out of the frame.
- Optional, slower but sometimes better pose: `pip install ultralytics`, download `yolo11m-pose.pt` into `models/`, then `--extractor yolo`.
