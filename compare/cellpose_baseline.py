"""Cellpose (instance segmentation) as a counting baseline — zero-shot, no training.

    python compare/cellpose_baseline.py

Runs the pretrained Cellpose `nuclei` model on the val split, derives a count
(number of mask labels) and bounding boxes (one per label), and scores the same
metrics as the detectors. Appends a "Cellpose (nuclei)" row to compare/results.json.

Cellpose commits to one segmentation and emits no per-object confidence, so:
  - mAP@50 is computed with every box scored 1.0 (its single operating point);
  - precision / recall / F1 @ IoU 0.5 (greedy match) is the honest apples-to-apples
    number vs. the detectors' tuned operating point.
"""
import json
import pathlib
import sys
import time

import cv2
import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset import YoloDetectionDataset  # noqa: E402


def masks_to_boxes(lbl):
    boxes = []
    for v in np.unique(lbl):
        if v == 0:
            continue
        ys, xs = np.where(lbl == v)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def pr_f1_at_iou(pred, gt, thr=0.5):
    """greedy 1-to-1 matching, IoU >= thr."""
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0
    pred, gt = torch.tensor(pred), torch.tensor(gt)
    from torchvision.ops import box_iou
    iou = box_iou(pred, gt).numpy()
    matched_gt, tp = set(), 0
    for pi in np.argsort(-iou.max(axis=1)):
        gi = int(iou[pi].argmax())
        if iou[pi, gi] >= thr and gi not in matched_gt:
            matched_gt.add(gi)
            tp += 1
    prec = tp / len(pred)
    rec = tp / len(gt)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def main():
    from cellpose import models
    ds = YoloDetectionDataset("dataset", "val", train=False)
    model = models.Cellpose(gpu=torch.cuda.is_available(), model_type="nuclei")

    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy",
                                  max_detection_thresholds=[10, 100, 500])
    metric.warn_on_many_detections = False
    gt_c, pred_c, precs, recs, f1s = [], [], [], [], []
    t0 = time.time()
    for i in range(len(ds)):
        img = cv2.imread(str(ds.imgs[i]), cv2.IMREAD_UNCHANGED)
        gray = img[..., 1] if img.ndim == 3 else img          # green channel = signal
        lbl, _, _, _ = model.eval(gray, diameter=None, channels=[0, 0])
        boxes = masks_to_boxes(lbl)
        _, target = ds[i]
        gt = target["boxes"].numpy()

        metric.update(
            [{"boxes": torch.tensor(boxes), "scores": torch.ones(len(boxes)),
              "labels": torch.ones(len(boxes), dtype=torch.int64)}],
            [{"boxes": torch.tensor(gt), "labels": target["labels"]}],
        )
        gt_c.append(len(gt))
        pred_c.append(int(lbl.max()))
        p, r, f = pr_f1_at_iou(boxes, gt, 0.5)
        precs.append(p); recs.append(r); f1s.append(f)
    dt = (time.time() - t0) / len(ds) * 1000

    m = metric.compute()
    gt_a, pc_a = np.array(gt_c), np.array(pred_c)
    ae = np.abs(pc_a - gt_a)
    nz = gt_a > 0
    row = {
        "mAP50": round(float(m["map_50"]), 4),
        "mAP75": round(float(m["map_75"]), 4),
        "mAR500": round(float(m["mar_500"]), 4),
        "P@0.5": round(float(np.mean(precs)), 4),
        "R@0.5": round(float(np.mean(recs)), 4),
        "F1@0.5": round(float(np.mean(f1s)), 4),
        "count_MAE@0.5": round(float(ae.mean()), 2),
        "count_MAPE@0.5_pct": round(float((ae[nz] / gt_a[nz]).mean() * 100), 2),
        "count_bias@0.5": int((pc_a - gt_a).sum()),
        "best_conf": None,
        "count_MAE@best": round(float(ae.mean()), 2),
        "count_MAPE@best_pct": round(float((ae[nz] / gt_a[nz]).mean() * 100), 2),
        "ms_per_image": round(dt, 1),
    }

    path = pathlib.Path("compare/results.json")
    results = json.loads(path.read_text()) if path.exists() else {}
    results["Cellpose (nuclei)"] = row
    path.write_text(json.dumps(results, indent=2))

    print("\nCellpose (nuclei), zero-shot, val split:")
    for k, v in row.items():
        print(f"  {k:<20} {v}")
    print("\nappended -> compare/results.json")


if __name__ == "__main__":
    main()
