"""Follow-up D — how much of the ~0.10 F1@0.5 residual (§2.4) is model error vs.
ground-truth (watershed box) error?

    python compare/followup_d.py

No human annotator available, so we triangulate the GT with two independent
nucleus labellers on the official test split (50 images):

  watershed  : current GT — watershed of BBBC039's curated semantic masks
  cellpose   : a Cellpose model fine-tuned ONLY on the official 100-image train
               split (never saw the test images) -> independent instance masks
  consensus  : nuclei both methods agree on (matched box IoU >= 0.5)

Then re-score YOLOv8s / RT-DETR-L F1@0.5 against each. If model F1 jumps toward
~1.0 on the consensus set, the residual was label ambiguity, not model error.
Also samples SAM (ultralytics, everything-mode) on 12 images as a noisy third
opinion on the raw count.
"""
import json
import pathlib
import sys

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "bbbc039"))
from dataset import YoloDetectionDataset  # noqa: E402
from eval_all import pr_f1_at_iou, rtdetr_preds, yolo_preds  # noqa: E402

CELLPOSE_FT = "compare/runs/cellpose_ft_official/models/bbbc039_ft"


def masks_to_boxes(lbl):
    b = []
    for v in np.unique(lbl):
        if v == 0:
            continue
        ys, xs = np.where(lbl == v)
        if xs.size >= 20:
            b.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(b, np.float32).reshape(-1, 4)


def match(a, b, thr=0.5):
    """indices of a,b that match at IoU>=thr (Hungarian)."""
    if len(a) == 0 or len(b) == 0:
        return np.array([], int), np.array([], int)
    iou = box_iou(torch.as_tensor(a, dtype=torch.float32),
                  torch.as_tensor(b, dtype=torch.float32)).numpy()
    ri, ci = linear_sum_assignment(-iou)
    ok = iou[ri, ci] >= thr
    return ri[ok], ci[ok]


def cellpose_gt(ds):
    from cellpose import models
    m = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=CELLPOSE_FT)
    out = []
    for i in range(len(ds)):
        g = cv2.imread(str(ds.imgs[i]), cv2.IMREAD_UNCHANGED)
        g = g[..., 1] if g.ndim == 3 else g
        lbl = m.eval(g, diameter=None, channels=[0, 0])[0]
        out.append(masks_to_boxes(lbl))
    return out


def sam_counts(ds, n=12):
    from ultralytics import SAM
    m = SAM("mobile_sam.pt")
    res = []
    for i in range(0, len(ds), max(1, len(ds) // n))[:n]:
        r = m.predict(str(ds.imgs[i]), verbose=False)[0]
        if r.masks is None:
            res.append((i, 0)); continue
        areas = r.masks.data.sum((1, 2)).cpu().numpy()
        res.append((i, int(((areas > 60) & (areas < 4000)).sum())))  # nucleus-sized only
    return res


def main():
    ds = YoloDetectionDataset("dataset_official", "test", train=False)
    ws = [ds[i][1]["boxes"].numpy() for i in range(len(ds))]
    print(f"test: {len(ds)} images, {sum(len(w) for w in ws)} watershed-GT nuclei\n")

    print("running Cellpose (official-train only) ...", flush=True)
    cp = cellpose_gt(ds)

    # GT agreement
    tp = fp_ws = fp_cp = 0
    consensus = []
    for w, c in zip(ws, cp):
        wi, ci = match(w, c)
        tp += len(wi)
        fp_ws += len(w) - len(wi)
        fp_cp += len(c) - len(ci)
        cons = (w[wi] + c[ci]) / 2 if len(wi) else np.zeros((0, 4), np.float32)
        consensus.append(cons.astype(np.float32))
    n_ws, n_cp, n_con = sum(map(len, ws)), sum(map(len, cp)), sum(map(len, consensus))
    print(f"\nGT counts:  watershed {n_ws}   cellpose {n_cp}   consensus {n_con}")
    print(f"  watershed-only (not in cellpose): {fp_ws} ({fp_ws/n_ws*100:.1f}%)")
    print(f"  cellpose-only  (not in watershed): {fp_cp} ({fp_cp/n_cp*100:.1f}%)")

    # SAM sanity
    try:
        sam = sam_counts(ds)
        sam_mae = np.mean([abs(k - len(ws[i])) for i, k in sam])
        print(f"  SAM everything-mode (n={len(sam)}): mean |SAM count - watershed| = {sam_mae:.1f}")
    except Exception as e:
        print(f"  SAM skipped: {e}")

    # re-score models vs each GT
    out = {"gt_counts": {"watershed": n_ws, "cellpose": n_cp, "consensus": n_con},
           "gt_disagreement": {"watershed_only_pct": round(fp_ws / n_ws * 100, 1),
                               "cellpose_only_pct": round(fp_cp / n_cp * 100, 1)},
           "F1@0.5": {}}
    models = {
        "YOLOv8s": lambda: yolo_preds("runs/detect/runs/yolo_s0/weights/best.pt", ds),
        "RT-DETR-L": lambda: rtdetr_preds("runs/detect/compare/runs/rtdetr_s0/weights/best.pt", ds),
    }
    for label, fn in models.items():
        preds = list(fn())
        row = {}
        for gname, G in (("watershed", ws), ("cellpose", cp), ("consensus", consensus)):
            # sweep conf, report best-F1 (not best-count) so it's a clean detection number
            best = 0.0
            for c in np.round(np.arange(0.2, 0.75, 0.05), 2):
                f1 = np.mean([pr_f1_at_iou(p["boxes"].numpy()[p["scores"].numpy() >= c], g, 0.5)[2]
                              for p, g in zip(preds, G)])
                best = max(best, float(f1))
            row[gname] = round(best, 4)
        out["F1@0.5"][label] = row
        print(f"\n{label} F1@0.5 (best over conf):  "
              f"vs watershed {row['watershed']:.3f}   vs cellpose {row['cellpose']:.3f}   "
              f"vs consensus {row['consensus']:.3f}")

    pathlib.Path("compare/followup_d.json").write_text(json.dumps(out, indent=2))
    print("\nsaved -> compare/followup_d.json")


if __name__ == "__main__":
    main()
