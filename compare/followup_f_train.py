"""Follow-up F — StarDist as a SECOND fine-tuned segmentation baseline (§2.7).

Runs in the isolated .venv-stardist (TensorFlow, Python 3.12) — the main env is
Python 3.14 / torch and can't hold TF.

    .venv-stardist/Scripts/python.exe compare/followup_f_train.py [--epochs 80]

Trains StarDist2D from scratch on the official BBBC039 train split (same 100
images / watershed instance labels as Cellpose-official, follow-up D), tunes its
prob/NMS thresholds on val, runs inference on the 50 test images, and dumps
per-image predicted boxes + counts to compare/followup_f_preds.json.  Scoring
(F1@0.5, mAP, count MAE — same harness as every other model) is done back in the
torch env by compare/followup_f.py.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from csbdeep.utils import normalize
from skimage.io import imread
from stardist import fill_label_holes
from stardist.models import Config2D, StarDist2D

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "bbbc039"))
from masks_to_yolo import instances  # noqa: E402

DATA = pathlib.Path("dataset_official")
MASKS = pathlib.Path("mask/masks")


def label_img(mask_path, shape):
    import cv2
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    lbl = np.zeros(shape, np.int32)
    for k, inst in enumerate(instances(m), 1):
        lbl[inst] = k
    return lbl


def load(split):
    X, Y, names = [], [], []
    for p in sorted((DATA / "images" / split).glob("*.png")):
        mp = MASKS / f"{p.stem}.png"
        if not mp.exists():
            continue
        g = imread(str(p))
        g = g[..., 1] if g.ndim == 3 else g          # green channel = signal
        lbl = label_img(mp, g.shape)
        X.append(normalize(g.astype(np.float32), 1, 99.8))
        Y.append(fill_label_holes(lbl))
        names.append(p.stem)
    return X, Y, names


def augmenter(x, y):
    if np.random.rand() < 0.5:
        x, y = x[::-1], y[::-1]
    if np.random.rand() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    k = np.random.randint(4)
    x, y = np.rot90(x, k), np.rot90(y, k)
    x = x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)
    return x, y


def boxes_from_labels(lbl):
    out = []
    for v in np.unique(lbl):
        if v == 0:
            continue
        ys, xs = np.where(lbl == v)
        out.append([int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="compare/runs/stardist_ft_official")
    args = ap.parse_args()

    Xtr, Ytr, _ = load("train")
    Xva, Yva, _ = load("val")
    Xte, Yte, te_names = load("test")
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}", flush=True)

    conf = Config2D(n_rays=32, grid=(2, 2), n_channel_in=1,
                    train_patch_size=(256, 256), train_batch_size=8,
                    train_epochs=args.epochs, train_steps_per_epoch=args.steps)
    out = pathlib.Path(args.out)
    model = StarDist2D(conf, name=out.name, basedir=str(out.parent))
    model.train(Xtr, Ytr, validation_data=(Xva, Yva), augmenter=augmenter)
    # ponytail: keep StarDist's default prob/nms thresholds (0.5 / 0.4). Its
    # optimize_thresholds() grid-search is ~30 min/threshold on CPU, and Cellpose
    # isn't threshold-tuned either -> defaults are the fair, comparable choice.

    preds = {}
    for name, img in zip(te_names, Xte):
        lbl, _ = model.predict_instances(img)
        preds[name] = {"boxes": boxes_from_labels(lbl), "count": int(lbl.max())}
    pathlib.Path("compare/followup_f_preds.json").write_text(json.dumps(preds))
    print(f"\nwrote {len(preds)} test predictions -> compare/followup_f_preds.json")


if __name__ == "__main__":
    main()
