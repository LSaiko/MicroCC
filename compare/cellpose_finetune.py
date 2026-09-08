"""Fine-tune Cellpose on BBBC039 — the fair trained-vs-trained segmentation baseline.

    python compare/cellpose_finetune.py [--epochs 150]

Builds instance-label masks from BBBC039's semantic masks (interior-class
connected components, same split as everything else), fine-tunes the pretrained
`nuclei` model on the 160 train images, saves to compare/runs/cellpose_ft/.
Then score it with:
    python compare/cellpose_baseline.py --model <printed path> --label "Cellpose (finetuned)"
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bbbc039"))
from masks_to_yolo import instances  # noqa: E402


def instance_mask(mask_path):
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    lbl = np.zeros(m.shape[:2], np.int32)
    for k, inst in enumerate(instances(m), 1):
        lbl[inst] = k
    return lbl


def load_split(split, masks_dir):
    img_dir = pathlib.Path("dataset/images") / split
    imgs, labels = [], []
    for p in sorted(img_dir.glob("*.png")):
        mp = masks_dir / f"{p.stem}.png"
        if not mp.exists():
            continue
        g = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        g = g[..., 1] if g.ndim == 3 else g
        lbl = instance_mask(mp)
        if lbl.max() < 5:                       # cellpose min_train_masks
            continue
        imgs.append(g)
        labels.append(lbl)
    return imgs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", type=pathlib.Path, default=pathlib.Path("mask/masks"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.0005)
    args = ap.parse_args()

    from cellpose import models, train

    tr_x, tr_y = load_split("train", args.masks)
    va_x, va_y = load_split("val", args.masks)
    print(f"train {len(tr_x)} imgs, val {len(va_x)} imgs")

    out = pathlib.Path("compare/runs/cellpose_ft")
    out.mkdir(parents=True, exist_ok=True)
    model = models.CellposeModel(gpu=torch.cuda.is_available(), model_type="nuclei")

    ret = train.train_seg(
        model.net,
        train_data=tr_x, train_labels=tr_y,
        test_data=va_x, test_labels=va_y,
        channels=[0, 0], normalize=True,
        n_epochs=args.epochs, learning_rate=args.lr, weight_decay=1e-4,
        save_path=str(out), model_name="bbbc039_ft",
    )
    path = str(ret[0] if isinstance(ret, tuple) else ret)   # 3.1.x returns (path, losses)
    print(f"\nsaved fine-tuned model -> {path}")
    print(f'score it:  python compare/cellpose_baseline.py --model "{path}" --label "Cellpose (finetuned)"')


if __name__ == "__main__":
    main()
