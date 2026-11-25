"""M1 leave-one-group-out loop: window features -> classifier -> alarm rule (thr, k), all picked without the test group.

Outer loop: train on 4 groups of annotations/subjects.csv, test on the 5th. The alarm rule (probability > thr
for k consecutive frames) is chosen on the 4 training groups from inner out-of-fold predictions.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import pose, urfd, windows
from .eval import evaluate

GRID = [(t, k) for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for k in [3, 6, 10, 15]]


def load_groups():
    return pd.read_csv(urfd.ROOT / "annotations" / "subjects.csv").set_index("seq").group


def build(extractor, win_s=1.0, onsets=None, signals="full", n_aug=0, seed=0):
    """Window features X, targets Y and train masks USE per sequence.

    With n_aug > 0 also returns AUG {seq: [(X, Y, USE), ...]}: copies of each sequence seen by a randomly
    transformed camera (windows.augment), used for training only."""
    onsets = urfd.fall_onsets() if onsets is None else onsets
    rng = np.random.default_rng(seed)
    X, Y, USE, AUG = {}, {}, {}, {}
    for s in urfd.sequences():
        kp, box = pose.load(extractor, s)
        X[s] = windows.window_features(windows.frame_signals(kp, box, signals=signals), win_s)
        Y[s], USE[s] = windows.targets(s, len(X[s]), onsets)
        AUG[s] = []
        for _ in range(n_aug):
            k2, b2 = windows.augment(kp, box, rng)
            AUG[s].append((windows.window_features(windows.frame_signals(k2, b2, signals=signals), win_s), Y[s], USE[s]))
    return (X, Y, USE, AUG) if n_aug else (X, Y, USE)


def to_results(proba, rule):
    t, k = rule
    return {s: (["STANDING"] * len(p), windows.alarms_from_proba(p, t, k)) for s, p in proba.items()}


def score(res, onsets, lab):
    d = evaluate(res, onsets, lab)[0]
    return d.hit.fillna(False).astype(int).sum() - d.false_alarms.sum()


def fit_on(model, train, X, Y, USE, aug=None):
    parts = [(X[s], Y[s], USE[s]) for s in train] + [a for s in train for a in (aug or {}).get(s, [])]
    Xt = pd.concat([x[u] for x, _, u in parts])
    yt = np.concatenate([y[u] for _, y, u in parts])
    return clone(model).fit(Xt, yt)


def pick_rule(model, X, Y, USE, groups, train, onsets, lab, aug=None):
    """Alarm rule (thr, k) from inner leave-one-group-out predictions on the training sequences."""
    oof = {}
    for h in sorted(groups[train].unique()):
        m = fit_on(model, [s for s in train if groups[s] != h], X, Y, USE, aug)
        oof.update({s: m.predict_proba(X[s])[:, 1] for s in train if groups[s] == h})
    return max(GRID, key=lambda r: score(to_results(oof, r), onsets, lab))


def loso(model, X, Y, USE, groups, onsets, lab, aug=None):
    """Returns (merged results {seq: (states, alarms)}, per-fold table, test probabilities {seq: proba})."""
    seqs = list(X)
    fit = lambda train: fit_on(model, train, X, Y, USE, aug)

    merged, rows, probas = {}, [], {}
    for g in sorted(groups.unique()):
        train = [s for s in seqs if groups[s] != g]
        test = [s for s in seqs if groups[s] == g]
        best = pick_rule(model, X, Y, USE, groups, train, onsets, lab, aug)
        m = fit(train)
        p = {s: m.predict_proba(X[s])[:, 1] for s in test}
        probas.update(p)
        res = to_results(p, best)
        merged.update(res)
        _, sm = evaluate(res, onsets, lab)
        rows.append(dict(group=g, thr=best[0], k=best[1], falls_hit=sm["falls_hit"], recall=sm["event_recall"],
                         adl_fa=sm["adl_false_alarms"]))
    return merged, pd.DataFrame(rows), probas
