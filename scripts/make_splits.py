"""Build the sequence table and the fixed cross-validation splits.

Usage: python scripts/make_splits.py

Writes (tracked in git, so every model is scored on the same folds):
  splits/sequences.csv  one row per sequence: seq, kind, label (fall=1, adl=0),
                        n_frames (RGB frames), n_labeled (rows in the label csv)
  splits/splits.json    5 stratified folds x 3 seeds; whole sequences only,
                        6 falls + 8 ADLs in every test fold

Needs data/raw/urfd/ from scripts/download_urfd.py (labels and rgb/).
Re-running gives identical files.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "urfd"
OUT = REPO / "splits"
N_FOLDS = 5
SEEDS = [0, 1, 2]


def sequence_table():
    lab = pd.concat([pd.read_csv(RAW / "labels" / f"urfall-cam0-{k}.csv", header=None, usecols=[0])
                     for k in ["falls", "adls"]], ignore_index=True)
    n_labeled = lab[0].value_counts()
    rgb = RAW / "rgb"
    if not rgb.is_dir():
        raise SystemExit(f"{rgb} not found: run scripts/download_urfd.py first (with RGB frames).")
    seqs = sorted(p.name for p in rgb.iterdir() if p.is_dir())
    df = pd.DataFrame({"seq": seqs})
    df["kind"] = df.seq.str.split("-").str[0]
    df["label"] = (df.kind == "fall").astype(int)
    df["n_frames"] = [sum(1 for _ in (rgb / s).rglob("*.png")) for s in seqs]
    df["n_labeled"] = df.seq.map(n_labeled).fillna(0).astype(int)
    return df


def make_splits(df):
    out = {"n_folds": N_FOLDS, "seeds": SEEDS, "splits": []}
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        folds = [{"fold": f, "train": sorted(df.seq[tr]), "test": sorted(df.seq[te])}
                 for f, (tr, te) in enumerate(skf.split(df.seq, df.label))]
        out["splits"].append({"seed": seed, "folds": folds})
    return out


def check(df, splits):
    assert len(df) == 70 and df.label.sum() == 30, "expected 70 sequences (30 falls, 40 ADLs)"
    assert (df.n_frames > 0).all() and (df.n_labeled > 0).all(), "a sequence has no frames or no labels"
    kind = dict(zip(df.seq, df.kind))
    for s in splits["splits"]:
        tests = [q for f in s["folds"] for q in f["test"]]
        assert sorted(tests) == sorted(df.seq), f"seed {s['seed']}: not every sequence is tested exactly once"
        for f in s["folds"]:
            assert not set(f["train"]) & set(f["test"]), "train/test overlap"
            assert len(f["train"]) + len(f["test"]) == len(df)
            n_fall = sum(kind[q] == "fall" for q in f["test"])
            assert (n_fall, len(f["test"]) - n_fall) == (6, 8), f"seed {s['seed']} fold {f['fold']}: {n_fall} falls"


def main():
    df = sequence_table()
    splits = make_splits(df)
    check(df, splits)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "sequences.csv", index=False, lineterminator="\n")
    (OUT / "splits.json").write_text(json.dumps(splits, indent=1) + "\n", encoding="utf-8")
    print(f"{len(df)} sequences ({df.label.sum()} falls), {len(SEEDS)} seeds x {N_FOLDS} folds -> {OUT}")


if __name__ == "__main__":
    main()
