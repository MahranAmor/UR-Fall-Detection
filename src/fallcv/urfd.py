"""URFD (UR Fall Detection, camera 0) access: sequences, frames, labels and fall events."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
URFD = ROOT / "data" / "raw" / "urfd"
FPS = 30
FEATS = ["HeightWidthRatio", "MajorMinorRatio", "BBoxOccupancy", "MaxStdXZ", "HHmaxRatio", "H", "D", "P40"]


def sequences():
    return sorted(p.name for p in (URFD / "rgb").iterdir() if p.is_dir())


def frames(seq):
    """Sorted RGB frame paths; frame number = index + 1."""
    return sorted((URFD / "rgb" / seq).rglob("*.png"))


def labels():
    """Per-frame labels: -1 not lying, 0 transition (falling or lying down), 1 lying on the floor."""
    lab = pd.concat([pd.read_csv(URFD / "labels" / f"urfall-cam0-{k}.csv", header=None) for k in ["falls", "adls"]])
    lab.columns = ["seq", "frame", "label"] + FEATS
    return lab[["seq", "frame", "label"]].reset_index(drop=True)


def fall_onsets(lab=None):
    """Fall sequence -> onset frame (first transition frame, label 0)."""
    lab = labels() if lab is None else lab
    f = lab[lab.seq.str.startswith("fall") & (lab.label == 0)]
    return f.groupby("seq").frame.min().to_dict()
