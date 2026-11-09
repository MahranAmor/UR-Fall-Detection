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

## Dataset citation

B. Kwolek, M. Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014. License CC BY-NC-SA 4.0 (non-commercial academic use).
