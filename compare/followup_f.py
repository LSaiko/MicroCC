"""Follow-up F — score fine-tuned StarDist next to fine-tuned Cellpose (§2.7).

    python compare/followup_f.py

Answers "is fine-tuned Cellpose representative of segmentation, or a one-off?"
by putting a second, architecturally different trained segmentation model
(StarDist2D — star-convex polygon regression) through the same harness on the
official BBBC039 test split.

StarDist predictions come from compare/followup_f_preds.json, produced by
compare/followup_f_train.py in the .venv-stardist TF env.  Cellpose-official
(trained on official-train only, follow-up D) is run here directly.
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
from eval_all import pr_f1_at_iou  # noqa: E402

DS = YoloDetectionDataset("dataset_official", "test", train=False)
REF = {"YOLOv8s (3 seeds)": 0.937, "Faster R-CNN (3 seeds)": 0.943,
       "Cellpose finetuned (custom val, main study)": 0.903}


def masks_to_boxes(lbl):
    out = []
    for v in np.unique(lbl):
        if v == 0:
            continue
        ys, xs = np.where(lbl == v)
        out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(out, dtype=np.float32).reshape(-1, 4)


def score_seg(box_iter, count_iter):
    """box_iter/count_iter: per-image predicted boxes (Nx4) and integer counts,
    in DS order.  Single operating point, every box scored 1.0 (like Cellpose)."""
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy",
                                  max_detection_thresholds=[10, 100, 500])
    metric.warn_on_many_detections = False
    gt_c, pred_c, f1s, ps, rs = [], [], [], [], []
    t0 = time.time()
    for i, (boxes, cnt) in enumerate(zip(box_iter, count_iter)):
        _, tgt = DS[i]
        gt = tgt["boxes"].numpy()
        metric.update(
            [{"boxes": torch.tensor(boxes).reshape(-1, 4), "scores": torch.ones(len(boxes)),
              "labels": torch.ones(len(boxes), dtype=torch.int64)}],
            [{"boxes": torch.tensor(gt), "labels": tgt["labels"]}])
        p, r, f = pr_f1_at_iou(boxes, gt, 0.5)
        ps.append(p); rs.append(r); f1s.append(f)
        gt_c.append(len(gt)); pred_c.append(int(cnt))
    dt = (time.time() - t0) / len(DS) * 1000
    m = metric.compute()
    gt_a, pc_a = np.array(gt_c), np.array(pred_c)
    ae = np.abs(pc_a - gt_a)
    nz = gt_a > 0
    return {
        "mAP50": round(float(m["map_50"]), 4),
        "mAR500": round(float(m["mar_500"]), 4),
        "P@0.5": round(float(np.mean(ps)), 4),
        "R@0.5": round(float(np.mean(rs)), 4),
        "F1@0.5": round(float(np.mean(f1s)), 4),
        "count_MAE": round(float(ae.mean()), 2),
        "count_MAPE_pct": round(float((ae[nz] / gt_a[nz]).mean() * 100), 2),
        "count_bias": int((pc_a - gt_a).sum()),
        "ms_per_image": round(dt, 1),
    }


def stardist_row():
    p = pathlib.Path("compare/followup_f_preds.json")
    if not p.exists():
        print("  ! followup_f_preds.json missing — run followup_f_train.py in .venv-stardist first")
        return None
    preds = json.loads(p.read_text())
    boxes = [np.array(preds[s.stem]["boxes"], dtype=np.float32).reshape(-1, 4) for s in DS.imgs]
    counts = [preds[s.stem]["count"] for s in DS.imgs]
    return score_seg(boxes, counts)


def cellpose_row():
    model_path = "compare/runs/cellpose_ft_official/models/bbbc039_ft"
    if not pathlib.Path(model_path).exists():
        print(f"  ! {model_path} missing")
        return None
    from cellpose import models
    m = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=model_path)
    boxes, counts = [], []
    for s in DS.imgs:
        img = cv2.imread(str(s), cv2.IMREAD_UNCHANGED)
        gray = img[..., 1] if img.ndim == 3 else img
        lbl = m.eval(gray, diameter=None, channels=[0, 0])[0]
        boxes.append(masks_to_boxes(lbl))
        counts.append(int(lbl.max()))
    return score_seg(boxes, counts)


def main():
    out = {"reference_F1@0.5": REF, "split": "dataset_official/test (50 imgs)", "rows": {}}
    for name, fn in (("StarDist (finetuned)", stardist_row),
                     ("Cellpose (finetuned, official)", cellpose_row)):
        print(f"scoring {name} ...", flush=True)
        r = fn()
        if r:
            out["rows"][name] = r

    pathlib.Path("compare/followup_f.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'model':<32} {'F1@0.5':>8} {'mAP@50':>8} {'count MAE':>10} {'MAPE%':>7}")
    for k, v in out["rows"].items():
        print(f"{k:<32} {v['F1@0.5']:>8.3f} {v['mAP50']:>8.3f} "
              f"{v['count_MAE']:>10.2f} {v['count_MAPE_pct']:>7.2f}")
    print(f"\nreference F1@0.5: " + "  ".join(f"{k.split('(')[0].strip()} {v}" for k, v in REF.items()))
    print("saved -> compare/followup_f.json")


if __name__ == "__main__":
    main()
