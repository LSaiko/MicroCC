"""Follow-up C — how much of RT-DETR-L's recall lead is "NMS-free", and how much
is just NMS being tuned wrong?

    python compare/followup_c.py

RT-DETR-L (ultralytics) is genuinely NMS-free (verified: its postprocess never
calls non_max_suppression). The anchor-based dense detectors (RetinaNet, FCOS)
run NMS with a fixed IoU threshold. With their per-image caps already raised
(§2.2), NMS is the last remaining recall bottleneck. We:

  1. sweep each model's NMS IoU threshold (0.30 strict -> 0.99 ~off) and measure
     F1@0.5 / recall / count MAE on the BBBC039 official test split;
  2. directly count GT nuclei that HAD an IoU>=0.5 match before NMS and LOST it
     after NMS (the "NMS suppression loss");
  3. compare the best-NMS anchor-based result to NMS-free RT-DETR-L.

Answers REPORT.md Sec 2.8.
"""
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset import YoloDetectionDataset  # noqa: E402
from eval_all import pr_f1_at_iou  # noqa: E402

DATA, SPLIT = "dataset_official", "test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NMS_SWEEP = [0.30, 0.45, 0.60, 0.75, 0.90, 0.99]
CONF_SWEEP = np.round(np.arange(0.2, 0.8, 0.05), 2)
FIXED_CONF = 0.5  # for the pre/post-NMS suppression measurement


def recall_at_iou(pred_boxes, gt_boxes, thr=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    iou = box_iou(torch.as_tensor(pred_boxes, dtype=torch.float32),
                  torch.as_tensor(gt_boxes, dtype=torch.float32)).numpy()
    ri, ci = linear_sum_assignment(-iou)
    return int((iou[ri, ci] >= thr).sum()) / len(gt_boxes)


def tv_model(arch):
    from train_tv import build
    ck = torch.load(f"compare/runs/{arch}/best.pt", map_location=DEVICE, weights_only=False)
    m = build(arch, ck.get("imgsz") or None).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.score_thresh = 0.01
    m.detections_per_img = 5000   # well above any per-image nucleus count; NMS IoU is the variable
    m.topk_candidates = 5000
    return m


def raw_boxes_no_nms(model, img):
    """RetinaNet/FCOS forward, decode, filter by score, skip NMS. detections cap
    lifted high so truncation (not suppression) isn't what removes boxes."""
    model.nms_thresh = 1.01           # batched_nms keeps everything
    with torch.no_grad():
        out = model([img.to(DEVICE)])[0]
    return out["boxes"].cpu().numpy(), out["scores"].cpu().numpy()


def eval_tv(arch, ds):
    model = tv_model(arch)
    gts = [ds[i][1]["boxes"].numpy() for i in range(len(ds))]

    # --- (2) pre/post-NMS suppression loss at a fixed conf ---
    pre_r, post_r = [], []
    model.nms_thresh = 0.5  # default
    for i in range(len(ds)):
        img, _ = ds[i]
        pb, ps = raw_boxes_no_nms(model, img)         # NMS off
        keep = ps >= FIXED_CONF
        pre_r.append(recall_at_iou(pb[keep], gts[i]))
        model.nms_thresh = 0.5
        with torch.no_grad():
            o = model([img.to(DEVICE)])[0]
        b, s = o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()
        post_r.append(recall_at_iou(b[s >= FIXED_CONF], gts[i]))
    supp = {"recall_no_nms": round(float(np.mean(pre_r)), 4),
            "recall_nms0.5": round(float(np.mean(post_r)), 4),
            "suppression_loss": round(float(np.mean(pre_r) - np.mean(post_r)), 4)}

    # --- (1) NMS IoU sweep ---
    curve = {}
    for nms in NMS_SWEEP:
        model.nms_thresh = nms
        preds = []
        for i in range(len(ds)):
            img, _ = ds[i]
            with torch.no_grad():
                o = model([img.to(DEVICE)])[0]
            preds.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()))

        def stats_at(c):
            f1 = np.mean([pr_f1_at_iou(b[s >= c], g, 0.5)[2] for (b, s), g in zip(preds, gts)])
            rec = np.mean([recall_at_iou(b[s >= c], g) for (b, s), g in zip(preds, gts)])
            pc = np.array([(s >= c).sum() for b, s in preds])
            gc = np.array([len(g) for g in gts])
            return float(f1), float(rec), float(np.abs(pc - gc).mean())

        best_c = min(CONF_SWEEP, key=lambda c: stats_at(c)[2])
        f1, rec, mae = stats_at(best_c)
        curve[f"{nms:.2f}"] = {"best_conf": round(float(best_c), 2),
                               "F1@0.5": round(f1, 4), "recall@0.5": round(rec, 4),
                               "count_MAE": round(mae, 2)}
    return {"suppression": supp, "nms_sweep": curve}


def eval_rtdetr(ds):
    from ultralytics import RTDETR
    m = RTDETR("runs/detect/compare/runs/rtdetr_s0/weights/best.pt")
    gts = [ds[i][1]["boxes"].numpy() for i in range(len(ds))]
    preds = []
    for i in range(len(ds)):
        r = m.predict(str(ds.imgs[i]), conf=0.05, iou=0.7, imgsz=960,
                      max_det=1000, verbose=False)[0].boxes
        preds.append((r.xyxy.cpu().numpy(), r.conf.cpu().numpy()))

    def stats_at(c):
        f1 = np.mean([pr_f1_at_iou(b[s >= c], g, 0.5)[2] for (b, s), g in zip(preds, gts)])
        rec = np.mean([recall_at_iou(b[s >= c], g) for (b, s), g in zip(preds, gts)])
        pc = np.array([(s >= c).sum() for b, s in preds])
        gc = np.array([len(g) for g in gts])
        return float(f1), float(rec), float(np.abs(pc - gc).mean())

    best_c = min(CONF_SWEEP, key=lambda c: stats_at(c)[2])
    f1, rec, mae = stats_at(best_c)
    return {"best_conf": round(float(best_c), 2), "F1@0.5": round(f1, 4),
            "recall@0.5": round(rec, 4), "count_MAE": round(mae, 2), "nms": "none (set prediction)"}


def main():
    ds = YoloDetectionDataset(DATA, SPLIT, train=False)
    print(f"{SPLIT} split: {len(ds)} images\n")

    out = {"RT-DETR-L (NMS-free)": eval_rtdetr(ds)}
    print("RT-DETR-L:", out["RT-DETR-L (NMS-free)"], "\n", flush=True)
    for arch in ("retinanet", "fcos"):
        print(f"scoring {arch} (NMS sweep) ...", flush=True)
        out[arch] = eval_tv(arch, ds)

    pathlib.Path("compare/followup_c.json").write_text(json.dumps(out, indent=2))

    rt = out["RT-DETR-L (NMS-free)"]
    print(f"\nRT-DETR-L (NMS-free):  F1@0.5 {rt['F1@0.5']:.3f}  recall {rt['recall@0.5']:.3f}  "
          f"count MAE {rt['count_MAE']:.2f}")
    for arch in ("retinanet", "fcos"):
        s = out[arch]["suppression"]
        print(f"\n{arch}:")
        print(f"  NMS suppression loss @ conf {FIXED_CONF}: recall {s['recall_no_nms']:.3f} (no NMS) "
              f"-> {s['recall_nms0.5']:.3f} (NMS 0.5)  = -{s['suppression_loss']:.3f}")
        print(f"  {'NMS IoU':>8} {'conf':>6} {'F1@0.5':>8} {'recall':>8} {'count MAE':>10}")
        for nms, v in out[arch]["nms_sweep"].items():
            print(f"  {nms:>8} {v['best_conf']:>6} {v['F1@0.5']:>8.3f} {v['recall@0.5']:>8.3f} "
                  f"{v['count_MAE']:>10.2f}")
    print("\nsaved -> compare/followup_c.json")


if __name__ == "__main__":
    main()
