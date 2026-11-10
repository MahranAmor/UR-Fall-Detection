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
curl -L -o models/efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite
```

The second file is a person detector (EfficientDet-Lite0, COCO). Skeleton extraction finds the person first and runs the pose model on a crop around them: on its own, the pose model sometimes puts a skeleton on a chair next to a person lying on the floor.

## Evaluation splits

```bash
python scripts/make_splits.py
```

Writes `splits/sequences.csv` (one row per sequence; label 1 = fall, 0 = ADL) and `splits/splits.json` (5 stratified folds x 3 seeds, 6 falls + 8 ADLs per test fold).
Both files are committed: every model must be trained and scored on these same folds, with whole sequences kept on one side of each split.

## Baseline: depth features (no images)

```bash
python scripts/baseline_depth.py
```

Summarises the 8 precomputed depth features per sequence and scores logistic regression and a random forest on the fixed splits (threshold 0.5).
Results go to `results/baseline_depth_summary.csv` and `results/baseline_depth_predictions.csv`.

| Model | Recall | Precision | F1 | AUC |
|---|---|---|---|---|
| Logistic regression, smoothed (best) | 1.000 | 0.971 | 0.985 | 0.999 |
| Logistic regression | 1.000 | 0.962 | 0.979 | 0.997 |
| Random forest | 1.000 | 0.905 | 0.947 | 0.994 |
| Control: sequence length only | 0.856 | 0.783 | 0.807 | 0.896 |

Mean over 3 seeds x 5 folds, per sequence. The only error of the best model is `adl-40` (a false alarm).
Note: at sequence level the task is easy, because every fall video starts upright and ends on the floor; the start-to-end change alone reaches F1 0.97.
The skeleton models should therefore also be compared at clip level (fixed 1.5 s windows), where that shortcut is not available.

## Skeletons (step 3)

```bash
python scripts/extract_skeletons.py --compare   # optional: compare settings on the 7 hardest sequences (~15 min)
python scripts/extract_skeletons.py             # extract all 70 sequences (~25 min, resumable)
```

Writes `data/processed/skeletons/<seq>.npz` (33 joints per frame; NaN where no person was kept; see the script docstring for the fields) and `results/skeleton_detection.csv`.

Method (`person_crop`): find the person with the person detector, run the pose model on a crop around them, retry with the crop turned 90 degrees, and only fall back to a full-frame pose if it is confident (visibility >= 0.8) or continues the previous one. Settings compared on the hardest sequences (`results/skeleton_settings_comparison.csv`):

| Setting | ADL lying | Fall lying | All frames | Skeletons on objects |
|---|---|---|---|---|
| IMAGE mode, conf 0.5 (exploration notebook) | 0.010 | 0.167 | 0.504 | yes |
| VIDEO mode, conf 0.3, rotation | 0.107 | 0.903 | 0.636 | yes (chair, backpack) |
| **person_crop (used)** | 0.121 | 0.778 | 0.626 | none found in visual checks |

A missing frame can be filled or flagged later; a skeleton on a chair labelled "lying" cannot be detected later, so the person-first method was chosen despite the lower raw rate.

Full run, all 70 sequences: 84.6% of frames have a skeleton; standing 87% (ADL) / 100% (fall), transition 97% / 98%, lying 67% / 85%.
Remaining misses are people curled up on the floor, in the dark (`adl-23`) or mostly out of frame (`adl-38`).
Joints MediaPipe cannot see get very low visibility (some leg joints are placed outside the image with visibility < 0.05), so later steps must treat low-visibility joints as missing.

## Cleaning (step 4)

```bash
python scripts/clean_skeletons.py
```

Writes `data/processed/clean/<seq>.npz` (joints in pixels, visibility, per-joint state observed / filled / missing, torso length) and `results/clean_summary.csv`. Each choice was measured on the data first:

| Operation | Setting | Why |
|---|---|---|
| Remove flips | hip > 0.5 torso lengths from the median of the 7 surrounding frames | curled-up bodies get fitted upside down for a few frames; impossible hip jumps 431 -> 15, all 30 falls keep their drop |
| Remove guesses | joints with visibility < 0.1 | about half of all leg joints are < 0.5 during falls but still usable, so only clear guesses go |
| Fill gaps | linear, gaps <= 5 frames | 93% of gaps are this short; longer ones (up to 100 frames) stay missing |
| Smooth | Savitzky-Golay, window 5 | standing jitter down to 40%, keeps 94% of the fall's peak speed (window 7: 88%, worst fall 53%) |

Centring and scaling are done per clip in step 5, not per sequence: people's apparent size changes 1.8x within a typical ADL as they walk (1.1x in falls), and in 2D a person lying along the camera axis cannot be told apart from an upright one, so a sequence-level scale either distorts ADLs or erases the fall.
Flip removal takes out 5.2% of fall frames but 0.5% of ADL frames, so missing frames themselves hint at the class: step 5 must treat gaps the same way for both.

## Clips (step 5)

```bash
python scripts/make_clips.py              # data/processed/clips.npz + results/clips_index.csv
python scripts/baseline_depth_clips.py    # depth baseline on the same clips
```

45-frame clips (1.5 s), a new one every 15 frames, every frame labeled. A clip is a **fall** if it holds at least 15 of the fall's 30 transition frames (at least 0.5 s of the fall); every ADL clip (including lying down on purpose) and fall-sequence clips before the fall are **not fall**; clips with only the edge of the fall or only lying afterwards are left out. Clips need at least 70% usable frames (at most 13 of 45 filled).
Result: **502 clips, 70 fall, 432 not fall.**

Each clip is normalised by its own first 10 usable frames: `pose` = joints relative to each frame's hip centre, `traj` = hip path relative to the start, both in reference torso lengths.

Shortcut checks on the same splits (a model given only this information):

| Information | AUC |
|---|---|
| share of missing / filled frames | 0.53 (chance) |
| typical skeleton shake | 0.50 (chance) |
| body size change in the clip | 0.68 (real fall motion: standing tall to lying) |

Depth baseline on the same clips (the bar for step 6):

| Model | Clip F1 | Clip AUC | Sequence recall | Sequence F1 |
|---|---|---|---|---|
| Random forest | 0.845 | 0.982 | 1.000 | 0.947 |
| Logistic regression | 0.777 | 0.973 | 1.000 | 0.782 |

Without the "ends on the floor" shortcut, the depth baseline drops from F1 0.985 per sequence to 0.845 per clip. Most false alarms are ADL clips of lying down on purpose.

## Dataset citation

B. Kwolek, M. Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014. License CC BY-NC-SA 4.0 (non-commercial academic use).
