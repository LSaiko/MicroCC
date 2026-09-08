"""Score every model through one metric harness on the val split.

    python compare/eval_all.py

Models:
  - YOLOv8s   : runs/detect/*/weights/best.pt  (via ultralytics)
  - FasterRCNN: compare/runs/fasterrcnn/best.pt (torchvision)
  - RetinaNet : compare/runs/retinanet/best.pt  (torchvision)

Metrics (identical for all): torchmetrics COCO mAP@50 / mAP@50:95 + count MAE / MAPE.
Writes compare/results.json and prints a table.
"""
import glob
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torchmetrics.detection import MeanAveragePrecision

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset import YoloDetectionDataset  # noqa: E402

CONF = 0.5          # matches evaluate.py; counting is measured at this threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ultralytics_preds(cls, weights, ds, imgsz):
    model = cls(weights)
    for i in range(len(ds)):
        r = model.predict(str(ds.imgs[i]), conf=0.001, iou=0.6, imgsz=imgsz,
                          max_det=1000, verbose=False)[0].boxes
        yield {"boxes": r.xyxy.cpu(), "scores": r.conf.cpu(),
               "labels": torch.ones(len(r), dtype=torch.int64)}


def yolo_preds(weights, ds):
    from ultralytics import YOLO
    return ultralytics_preds(YOLO, weights, ds, 1280)


def rtdetr_preds(weights, ds):
    from ultralytics import RTDETR
    return ultralytics_preds(RTDETR, weights, ds, 960)


def tv_preds(weights, ds, arch):
    from train_tv import build
    ckpt = torch.load(weights, map_location=DEVICE, weights_only=False)
    model = build(arch, ckpt.get("imgsz") or None).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # Uncap every COCO-tuned per-image limit: the defaults (100-300 detections,
    # 1000 pre-NMS candidates) are set for images with ~7 objects and silently
    # throttle recall on fields of 100-165 nuclei. Also drop the score floor so
    # the mAP PR-curve isn't truncated (YOLO uses 0.001).
    if arch == "fasterrcnn":
        model.roi_heads.score_thresh = 0.01
        model.roi_heads.detections_per_img = 1000
        model.rpn.post_nms_top_n_test = 3000
        model.rpn.pre_nms_top_n_test = 6000
    else:
        model.score_thresh = 0.01
        model.detections_per_img = 1000
        model.topk_candidates = 3000
    with torch.no_grad():
        for i in range(len(ds)):
            img, _ = ds[i]
            out = model([img.to(DEVICE)])[0]
            yield {k: v.cpu() for k, v in out.items()}


def score(pred_iter, ds):
    # nuclei fields hold up to ~165 objects. torchmetrics caps the 50:95 `map` at
    # 100 dets internally (penalises dense predictors), so we report mAP@50 / mAR
    # with a 500 cap -- the meaningful numbers for a counting task -- and skip 50:95.
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy",
                                  max_detection_thresholds=[10, 100, 500])
    metric.warn_on_many_detections = False
    sweep = np.round(np.arange(0.2, 0.85, 0.05), 2)
    gt_counts, scores_per_img = [], []
    t0 = time.time()
    for pred, i in zip(pred_iter, range(len(ds))):
        _, target = ds[i]
        metric.update([pred], [{"boxes": target["boxes"], "labels": target["labels"]}])
        gt_counts.append(len(target["boxes"]))
        scores_per_img.append(pred["scores"].numpy())
    dt = (time.time() - t0) / len(ds) * 1000
    m = metric.compute()
    gt = np.array(gt_counts)
    nz = gt > 0

    def stats_at(c):
        pc = np.array([(s >= c).sum() for s in scores_per_img])
        ae = np.abs(pc - gt)
        return (float(ae.mean()), int((pc - gt).sum()),
                float((ae[nz] / gt[nz]).mean() * 100))

    mae05, bias05, mape05 = stats_at(CONF)
    best_c = min(sweep, key=lambda c: stats_at(c)[0])
    mae_b, bias_b, mape_b = stats_at(best_c)
    return {
        "mAP50": round(float(m["map_50"]), 4),
        "mAP75": round(float(m["map_75"]), 4),
        "mAR500": round(float(m["mar_500"]), 4),
        "count_MAE@0.5": round(mae05, 2),
        "count_MAPE@0.5_pct": round(mape05, 2),
        "count_bias@0.5": bias05,
        "best_conf": round(float(best_c), 2),
        "count_MAE@best": round(mae_b, 2),
        "count_MAPE@best_pct": round(mape_b, 2),
        "ms_per_image": round(dt, 1),
    }


def main():
    ds = YoloDetectionDataset("dataset", "val", train=False)
    yolo_w = sorted((p for p in glob.glob("runs/detect/**/weights/best.pt", recursive=True)
                     if "rtdetr" not in p and "compare" not in p),
                    key=lambda p: pathlib.Path(p).stat().st_mtime)
    jobs = []
    if yolo_w:
        jobs.append(("YOLOv8s", lambda: yolo_preds(yolo_w[-1], ds)))
    rtdetr_w = next(iter(sorted(glob.glob("**/rtdetr*/weights/best.pt", recursive=True),
                                key=lambda p: pathlib.Path(p).stat().st_mtime, reverse=True)), None)
    if rtdetr_w:
        jobs.append(("RT-DETR-L", lambda: rtdetr_preds(rtdetr_w, ds)))
    for arch, label in (("fasterrcnn", "Faster R-CNN"), ("retinanet", "RetinaNet"),
                        ("fcos", "FCOS")):
        for w in sorted(glob.glob(f"compare/runs/{arch}*/best.pt")):
            suffix = pathlib.Path(w).parent.name.replace(arch, "").lstrip("_")
            lbl = f"{label} @{suffix}" if suffix else label
            jobs.append((lbl, (lambda a=arch, ww=w: tv_preds(ww, ds, a))))

    results = {}
    for label, fn in jobs:
        print(f"scoring {label} ...", flush=True)
        results[label] = score(fn(), ds)

    # merge, keeping any Cellpose rows written by cellpose_baseline.py
    path = pathlib.Path("compare/results.json")
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(results)
    path.write_text(json.dumps(merged, indent=2))

    cols = ["mAP50", "mAP75", "mAR500", "count_MAE@0.5", "best_conf",
            "count_MAE@best", "count_MAPE@best_pct", "ms_per_image"]
    print(f"\n{'model':<14} " + " ".join(f"{c:>18}" for c in cols))
    for label, r in results.items():
        print(f"{label:<14} " + " ".join(f"{r[c]:>18}" for c in cols))
    print("\nsaved -> compare/results.json")


if __name__ == "__main__":
    main()
