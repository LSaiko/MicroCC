"""Evaluate a trained YOLOv8 cell counter.

    python evaluate.py --model runs/detect/train/weights/best.pt \
                       --data DATASET/dataset.yaml --split val \
                       --json results.json

Reports mAP@50, mAP@50:95 (from ultralytics' validator) and counting error
(MAE / MAPE) computed by comparing #predicted boxes to #ground-truth boxes
per image. Prints a table and writes JSON.
"""
import argparse
import json
import pathlib

import numpy as np
import yaml
from ultralytics import YOLO


def count_lines(txt):
    if not txt.exists():
        return 0
    return sum(1 for l in txt.read_text().splitlines() if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, type=pathlib.Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.5)   # conf sweep: MAE minimises near 0.5-0.55
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--json", type=pathlib.Path, default=pathlib.Path("results.json"))
    args = ap.parse_args()

    model = YOLO(args.model)

    # --- detection metrics (ultralytics does the mAP math) ---
    m = model.val(data=str(args.data), split=args.split, conf=0.001,
                  iou=args.iou, imgsz=args.imgsz, verbose=False, workers=0)
    map50, map5095 = float(m.box.map50), float(m.box.map)

    # --- counting metrics ---
    cfg = yaml.safe_load(args.data.read_text())
    root = pathlib.Path(cfg["path"])
    img_dir = root / cfg[args.split]
    lbl_dir = pathlib.Path(str(img_dir).replace("images", "labels", 1))
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    if not imgs:
        raise SystemExit(f"no images in {img_dir}")

    gt, pred = [], []
    for p in imgs:
        gt.append(count_lines(lbl_dir / f"{p.stem}.txt"))
        r = model.predict(str(p), conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        pred.append(len(r.boxes))
    gt, pred = np.array(gt), np.array(pred)
    abs_err = np.abs(pred - gt)
    mae = float(abs_err.mean())
    nz = gt > 0
    mape = float((abs_err[nz] / gt[nz]).mean() * 100) if nz.any() else float("nan")

    results = {
        "model": str(args.model),
        "split": args.split,
        "n_images": len(imgs),
        "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "mAP50": round(map50, 4),
        "mAP50_95": round(map5095, 4),
        "count_MAE": round(mae, 3),
        "count_MAPE_pct": round(mape, 2),
        "gt_total": int(gt.sum()),
        "pred_total": int(pred.sum()),
        "bias_pred_minus_gt": int((pred - gt).sum()),
    }
    args.json.write_text(json.dumps(results, indent=2))

    w = 22
    print("\n" + "-" * 40)
    for k, v in results.items():
        print(f"{k:<{w}} {v}")
    print("-" * 40)
    print(f"saved -> {args.json}")


if __name__ == "__main__":
    main()
