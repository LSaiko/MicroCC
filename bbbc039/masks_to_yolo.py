"""BBBC039 nucleus masks -> YOLO bbox labels.

Usage:
    python masks_to_yolo.py --images IMAGES_DIR --masks MASKS_DIR --out LABELS_DIR

For every mask it writes LABELS_DIR/<stem>.txt with one line per nucleus:
    0 cx cy w h        (all normalised to 0-1)
Masks with no nuclei still get an empty .txt (YOLO treats it as a valid
background image).

BBBC039's downloadable masks are *semantic* RGBA PNGs: 0 = background,
1 = nucleus interior, 2 = boundary ring (3 = rare overlap). We recover true
instance masks by seeding on the interior class and **watershedding the full
foreground** (interior + boundary), so each box is the nucleus's real extent
(ring included) and touching nuclei are split along the ridge line even where
the boundary class has a gap. We also handle instance-labelled masks (many
unique values) and plain binary masks.
"""
import argparse
import pathlib

import cv2
import numpy as np

IMG_EXTS = (".tif", ".tiff", ".png")

# a BBBC039 nucleus is ~15-40 px across => ~150-1200 px^2; anything under this is
# an interior-class fragment, not a nucleus
MIN_AREA = 25
MIN_SIDE = 5


def to_label2d(mask):
    """Any TIFF/PNG(A) mask -> single-channel label array."""
    if mask.ndim == 3:
        mask = mask[..., :3].max(axis=2)  # only one colour channel carries the label
    return mask


def _watershed_instances(interior, foreground):
    from scipy import ndimage
    from skimage.segmentation import watershed
    # close hairline gaps so one nucleus isn't split into two seeds
    interior = cv2.morphologyEx(interior.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
    markers, _ = ndimage.label(interior)
    dist = ndimage.distance_transform_edt(foreground)
    ws = watershed(-dist, markers, mask=foreground)
    for v in range(1, ws.max() + 1):
        m = ws == v
        if m.sum():
            yield m


def instances(mask):
    """Yield boolean masks, one per nucleus."""
    lab = to_label2d(mask)
    vals = np.unique(lab)
    vals = vals[vals != 0]
    if vals.size == 0:
        return
    if vals.size > 4 and vals.max() > 8:                 # instance-labelled
        for v in vals:
            yield lab == v
    elif vals.max() <= 8 and vals.size <= 4:             # BBBC039 semantic
        yield from _watershed_instances(lab == vals.min(), lab > 0)
    else:                                                # plain binary
        n, cc = cv2.connectedComponents((lab > 0).astype(np.uint8))
        for v in range(1, n):
            yield cc == v


def mask_to_yolo_lines(mask, min_area=MIN_AREA, min_side=MIN_SIDE):
    h, w = to_label2d(mask).shape
    lines = []
    for inst in instances(mask):
        ys, xs = np.where(inst)
        if xs.size < min_area:                 # fragment, not a nucleus
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
        if bw < min_side or bh < min_side:
            continue
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
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--min-side", type=int, default=MIN_SIDE)
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
        lines = mask_to_yolo_lines(mask, args.min_area, args.min_side)
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
