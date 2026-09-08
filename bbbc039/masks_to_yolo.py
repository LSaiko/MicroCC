"""BBBC039 nucleus masks -> YOLO bbox labels.

Usage:
    python masks_to_yolo.py --images IMAGES_DIR --masks MASKS_DIR --out LABELS_DIR

For every mask it writes LABELS_DIR/<stem>.txt with one line per nucleus:
    0 cx cy w h        (all normalised to 0-1)
Masks with no nuclei still get an empty .txt (YOLO treats it as a valid
background image).

BBBC039's downloadable masks are *semantic* RGBA PNGs: 0 = background,
1 = nucleus interior, 2 = boundary ring (3 = rare overlap). Splitting on the
interior class turns the touching nuclei into separate connected components,
which is exactly the instance split we need. We also handle instance-labelled
masks (many unique values) and plain binary masks.
"""
import argparse
import pathlib

import cv2
import numpy as np

IMG_EXTS = (".tif", ".tiff", ".png")


def to_label2d(mask):
    """Any TIFF/PNG(A) mask -> single-channel label array."""
    if mask.ndim == 3:
        mask = mask[..., :3].max(axis=2)  # only one colour channel carries the label
    return mask


def instances(mask):
    """Yield boolean masks, one per nucleus."""
    lab = to_label2d(mask)
    vals = np.unique(lab)
    vals = vals[vals != 0]
    if vals.size == 0:
        return
    if vals.size > 4 and vals.max() > 8:            # instance-labelled
        for v in vals:
            yield lab == v
    else:                                          # semantic (interior class) or binary
        fg = (lab == vals.min()).astype(np.uint8)  # BBBC039 interior = 1; binary = 255
        n, cc = cv2.connectedComponents(fg)
        for v in range(1, n):
            yield cc == v


def mask_to_yolo_lines(mask, min_area=4, pad=1):
    h, w = to_label2d(mask).shape
    lines = []
    for inst in instances(mask):
        ys, xs = np.where(inst)
        if xs.size < min_area:  # ponytail: drop label noise / stray pixels
            continue
        x0 = max(xs.min() - pad, 0)
        x1 = min(xs.max() + pad, w - 1)
        y0 = max(ys.min() - pad, 0)
        y1 = min(ys.max() + pad, h - 1)          # pad recovers the ~1px boundary ring
        bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
        cx, cy = x0 + bw / 2, y0 + bh / 2
        lines.append(f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def find_mask(stem, masks_dir):
    for ext in (".png", ".tif", ".tiff"):
        p = masks_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=pathlib.Path)
    ap.add_argument("--masks", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--pad", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in sorted(args.images.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise SystemExit(f"no images in {args.images}")

    n_ok = n_empty = n_missing = total_boxes = 0
    for img in imgs:
        mpath = find_mask(img.stem, args.masks)
        if mpath is None:
            n_missing += 1
            print(f"  ! no mask for {img.name}")
            continue
        mask = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
        if mask is None:
            n_missing += 1
            print(f"  ! unreadable mask {mpath.name}")
            continue
        lines = mask_to_yolo_lines(mask, args.min_area, args.pad)
        (args.out / f"{img.stem}.txt").write_text("\n".join(lines))
        total_boxes += len(lines)
        if lines:
            n_ok += 1
        else:
            n_empty += 1

    print(f"done: {n_ok} labelled, {n_empty} empty, {n_missing} missing "
          f"({total_boxes} boxes total) -> {args.out}")


if __name__ == "__main__":
    main()
