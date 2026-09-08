"""Turn raw BBBC039 (TIFF images + label .txt from masks_to_yolo.py) into a
YOLO dataset: 3-channel PNGs, train/val split, dataset.yaml.

    python build_dataset.py --images IMAGES --labels LABELS --out DATASET [--val-frac 0.2]

Uses BBBC039's official train/val/test list files if you pass --splits DIR
(the folder with training.txt / validation.txt / test.txt); otherwise a
random split by --val-frac.
"""
import argparse
import pathlib
import random
import shutil

import cv2
import numpy as np


def fluoro_to_rgb(gray):
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = np.zeros((*norm.shape, 3), np.uint8)
    rgb[..., 1] = norm  # green false-colour; swap for a LUT if you want more contrast
    return rgb


def load_split_names(splits_dir):
    out = {}
    for name, fn in (("train", "training.txt"), ("val", "validation.txt")):
        p = splits_dir / fn
        if p.exists():
            out[name] = {pathlib.Path(l.strip()).stem for l in p.read_text().splitlines() if l.strip()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=pathlib.Path)
    ap.add_argument("--labels", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--splits", type=pathlib.Path, help="BBBC039 metadata dir with training.txt/validation.txt")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stems = sorted(p.stem for p in args.images.iterdir()
                   if p.suffix.lower() in (".tif", ".tiff", ".png"))
    stems = [s for s in stems if (args.labels / f"{s}.txt").exists()]
    if not stems:
        raise SystemExit("no image/label pairs found")

    if args.splits:
        named = load_split_names(args.splits)
        assign = {s: ("val" if s in named.get("val", set()) else "train") for s in stems
                  if s in named.get("train", set()) or s in named.get("val", set())}
        stems = list(assign)
    else:
        random.Random(args.seed).shuffle(stems)
        k = int(len(stems) * args.val_frac)
        assign = {s: "val" for s in stems[:k]}
        assign.update({s: "train" for s in stems[k:]})

    src_by_stem = {p.stem: p for p in args.images.iterdir()}
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for s, split in assign.items():
        gray = cv2.imread(str(src_by_stem[s]), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(args.out / "images" / split / f"{s}.png"), fluoro_to_rgb(gray))
        shutil.copyfile(args.labels / f"{s}.txt", args.out / "labels" / split / f"{s}.txt")

    (args.out / "dataset.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\n"
        "nc: 1\nnames: ['cell']\n"
    )
    n_val = sum(v == "val" for v in assign.values())
    print(f"wrote {len(assign)} images ({n_val} val) -> {args.out}/dataset.yaml")


if __name__ == "__main__":
    main()
