# Fall detection from video

**Goal:** take a video of a person and answer **fall** or **no fall**, and raise the alarm quickly when someone falls.

We compare two families of methods, as asked in the course:
- **Part A, traditional:** hand-made descriptors (HOG, LBP, colour histograms, SIFT) with classic classifiers (KNN, SVM, Random Forest);
- **Part B, deep learning:** CNNs on the images, and the body skeleton over time (rules, LSTM, graph network).

Then we test the best methods on **new cameras, rooms and people** (3 more datasets), choose the best one and turn it into a **demo program**.

**Result:** the **skeleton LSTM** is the best choice. It is first or second in every test, and on datasets it has never seen it gets **0.82 to 0.90** of the videos right. Methods that look at the image (colour, CNNs) score very well on URFD (up to 1.00), but when trained on URFD only they drop to chance level in new rooms.

---

## Project map

```
notebooks/   01 to 14: one notebook per step, each with an "Interpretation" under every result
scripts/     download_urfd.py: downloads the URFD dataset
             detect.py: the demo (fall / no fall for a video, a folder or a live camera)
models/      fall_lstm.pt: the final model (notebook 14); pose_landmarker_full.task: MediaPipe (to download)
results/     cross_dataset_mahran.csv: the scores of the cross-dataset test (notebook 13b)
             demo/: what the demo produces (videos, plots; not in git)
data/        the datasets and intermediate files (not in git, see "Data" below)
```

## The notebooks, step by step

| # | Notebook | What it does | Main result |
|---|---|---|---|
| 01 | `01_dataset_exploration` | looks at the URFD dataset | 70 videos: 30 falls, 40 daily activities (ADL) |
| 02 | `02_baseline_depth` | baseline with the depth features of URFD | 67 of 70 videos right; the score to beat |
| | **Part A: traditional** | | |
| 03 | `03_find_person` | finds the person (median background) and crops them | person found in 94% of the images, 100% of the fall images |
| 04 | `04_hog` | HOG + KNN / SVM / RF | video accuracy 0.857 (HOG + KNN) |
| 05 | `05_lbp` | LBP texture + KNN / SVM / RF | 0.757 to 0.786 on videos |
| 06 | `06_colour_histograms` | colour histograms + KNN / SVM / RF | best image scores of part A, but it may use the room |
| 07 | `07_sift` | SIFT + bag of visual words | dense SIFT 0.866 on images, weakest on videos |
| 08 | `08_traditional_benchmark` | compares all of part A, with a strict split (new people and rooms) | **a corner of the room alone gets 70 of 70**: the dataset has a shortcut |
| | **Part B: deep learning** | | |
| 09 | `09_skeletons` | MediaPipe skeletons (13 joints) | found in 82% of the images, less when lying |
| 10 | `10_state_machine` | rules over time: fast drop, lying, stays down | 30 of 30 falls, 6 false alarms, no training |
| 11 | `11_cnn` | small CNN, MobileNetV2, ResNet18, ResNet50 | better postures than part A, but they overfit and partly use something other than the person |
| 12 | `12_skeletons_over_time` | LSTM and graph network (GNN) on 1.5 s of skeleton | **LSTM 63 of 70 videos**, best on the strict split |
| | **New cameras, rooms, people** | | |
| 13a | `13a_new_datasets` | prepares URFD ceiling camera, GMDCSA-24 and CAUCAFall | 360 videos; the datasets mark the start of a fall differently, so we align it |
| 13b | `13b_cross_dataset` | trains on one or two datasets, tests on another one | **LSTM best on unseen datasets** (table below) |
| 14 | `14_final_model` | trains the final LSTM on all 330 videos and checks the demo | 0.92 with new people; saved in `models/fall_lstm.pt` |

## Main result: does it still work in a room it has never seen?

Video accuracy (a video counts as *fall* if there is an alarm anywhere in it). Answering *no fall* to everything gives 0.57 on URFD, 0.51 on GMDCSA and 0.50 on CAUCAFall.

| Method | URFD, new people | Trained on URFD, tested on GMDCSA / CAUCAFall | Trained on 2 datasets, tested on the 3rd: URFD / GMDCSA / CAUCAFall |
|---|---|---|---|
| **Skeleton LSTM** | 0.96 | **0.84 / 0.90** | **0.88 / 0.86 / 0.82** |
| State machine (rules) | 0.87 | 0.86 / 0.67 | 0.83 / 0.84 / 0.68 |
| Skeleton GNN | 0.96 | 0.76 / 0.68 | 0.84 / 0.85 / 0.69 |
| HOG + KNN | 0.84 | 0.78 / 0.55 | 0.60 / 0.64 / 0.54 |
| MobileNetV2 (CNN) | 0.89 | 0.51 / 0.50 | 0.77 / 0.61 / 0.66 |
| Colour HSV + SVM | **1.00** | 0.51 / 0.50 | 0.67 / 0.51 / 0.55 |

What it shows:
1. **One dataset is not enough to choose a method.** Colour + SVM is perfect on URFD and at chance level everywhere else: it learned the room, not the fall.
2. **The skeleton LSTM transfers best**, because it looks at how the body moves, not at the room.
3. **Limits:** the ceiling camera stays hard (at best 22 of 30 falls in time); lying on a bed, sleeping and exercising still cause false alarms; each dataset has only one to three rooms.

## The demo

The final model (notebook 14) is the mean of **3 skeleton LSTMs** trained on URFD, GMDCSA-24 and CAUCAFall. It is one file of 0.37 MB plus MediaPipe.
With people not used in training it gets **0.92** of the videos right: 143 of 159 falls caught in time, and 13 of 171 daily activities with a false alarm.

```bash
python scripts/detect.py my_video.mp4          # one video
python scripts/detect.py data/raw/home         # every video of a folder + results/demo/summary.csv
python scripts/detect.py --camera 0            # live, with the laptop camera (q = quit, r = reset the alarm)
```

It prints **FALL detected at … s** or **NO FALL**. In `results/demo/` it saves the video with the skeleton and the alarm drawn on it, and a plot of the fall score.

How it works: MediaPipe finds 13 joints in every image → we keep 10 images per second → the last 1.5 s of skeleton go into the LSTMs → fall score → alarm when the smoothed score goes above the threshold (0.989).

To record your own test videos: phone fixed, landscape, whole body visible, side view or high corner; **fall only onto a mattress**.

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

MediaPipe needs its pose model file:

```bash
mkdir -p models
curl -L -o models/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Data

`data/` is not in git: each of us downloads it locally.

**URFD** (notebooks 01 to 12):

```bash
python scripts/download_urfd.py            # everything (~4.4 GB)
python scripts/download_urfd.py --no-rgb   # skip the RGB frames (~100 MB)
```

This fetches the [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) (camera 0) into `data/raw/urfd/`:

| Folder | Content |
|---|---|
| `rgb/<seq>/` | 640x480 PNG frames, 30 fps (the input for pose estimation) |
| `videos/` | preview mp4s (depth and RGB side by side, 640x240) |
| `sync/` | per-frame timestamps and accelerometer magnitude |
| `labels/` | per-frame labels: -1 not lying, 0 transition, 1 lying on the floor (+ 8 depth features) |

70 sequences: `fall-01` to `fall-30` and `adl-01` to `adl-40` (daily activities, some include lying down on purpose). The script is safe to re-run: finished sequences are skipped.

**New test datasets** (notebooks 13a, 13b, 14), placed by hand:

| Folder | Dataset | Licence |
|---|---|---|
| `data/raw/urfd/rgb_cam1/` | URFD camera 1 (ceiling), the same 30 falls as camera 0 | CC BY-NC-SA 4.0 |
| `data/raw/gmdcsa24/` | [GMDCSA-24](https://doi.org/10.5281/zenodo.12921216): 160 videos, 4 people | MIT |
| `data/raw/caucafall/` | [CAUCAFall](https://doi.org/10.17632/7w7fccy7ky): 100 videos, 10 people | CC BY 4.0 |

**Order to run:** 01 → 14. Later notebooks use files saved by earlier ones in `data/processed/`. Notebook 13b loads its saved scores if they exist (the full training takes about 1 hour on a laptop GPU).

## Dataset citation

B. Kwolek, M. Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014. License CC BY-NC-SA 4.0 (non-commercial academic use).

GMDCSA-24 and CAUCAFall: see the links in the table above.
