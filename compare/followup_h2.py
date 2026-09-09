"""Follow-up H2 — a count-consistency loss *inside* YOLO training.

    python compare/followup_h2.py --lam 2.0 --epochs 150          # train
    python compare/followup_h2.py --eval runs/detect/runs/yolo_h2_s0/weights/best.pt

E and H showed neither a post-hoc threshold nor a threshold-free density model
reaches the per-image oracle (count MAE 2.13 -> 0.36). The oracle headroom lives
in the detector's own predictions, so the last lead is to change the training
objective: monkey-patch v8DetectionLoss to add

    lam * mean_b | soft_count_b - n_gt_b | / (n_gt_b + 1)

where soft_count_b is a differentiable NMS-free peak count of the P3 (stride-8)
class-score map: local maxima, weighted by their confidence. The term is a
*bounded* symmetric relative error (in [0, 1]) so a bad early batch can't spike
the gradient, and it is warmed up (off until --warmup epochs) so detection is
stable first. Answers REPORT.md §2.6 (follow-up H2).
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))

PEAK_K = 5      # local-max kernel on the P3 grid (~nucleus spacing at stride 8)
PEAK_FLOOR = 0.10
_STATE = {"epoch": 0, "warmup": 20}


def _soft_peak_count(p3_scores):
    """p3_scores: [B,1,H,W] sigmoid probs -> [B] differentiable peak count."""
    mx = F.max_pool2d(p3_scores, PEAK_K, stride=1, padding=PEAK_K // 2)
    is_peak = (p3_scores >= mx) & (p3_scores > PEAK_FLOOR)
    return (is_peak.float() * p3_scores).sum(dim=(1, 2, 3))


def patch_loss(lam):
    from ultralytics.utils.loss import v8DetectionLoss
    if getattr(v8DetectionLoss, "_h2_patched", False):
        return
    orig = v8DetectionLoss.__call__

    def patched(self, preds, batch):
        total, detached = orig(self, preds, batch)
        if _STATE["epoch"] < _STATE["warmup"]:
            return total, detached
        p = self.parse_output(preds)
        feats = p["feats"]
        b, _, h0, w0 = feats[0].shape
        n_p3 = h0 * w0
        s = p["scores"][:, :, :n_p3].reshape(b, self.nc, h0, w0).sigmoid()
        s = s.amax(dim=1, keepdim=True)  # class-max -> [b,1,h0,w0]
        soft_count = _soft_peak_count(s)
        n_gt = torch.bincount(batch["batch_idx"].long().flatten(),
                              minlength=b).float().to(soft_count.device)
        # bounded symmetric relative error in [0, 1] -> no gradient spikes
        count_loss = ((soft_count - n_gt).abs()
                      / (soft_count.detach() + n_gt + 1.0)).mean()
        return total + lam * count_loss * b, detached

    v8DetectionLoss.__call__ = patched
    v8DetectionLoss._h2_patched = True
    print(f"[h2] v8DetectionLoss patched: + {lam} * relative count error", flush=True)


def count_mae(weights):
    from dataset import YoloDetectionDataset
    from ultralytics import YOLO
    ds = YoloDetectionDataset("dataset_official", "test", train=False)
    m = YOLO(weights)
    gts = np.array([len(ds[i][1]["boxes"]) for i in range(len(ds))])
    score_lists = []
    for i in range(len(ds)):
        r = m.predict(str(ds.imgs[i]), conf=0.03, iou=0.6, imgsz=1280,
                      max_det=2000, verbose=False)[0]
        score_lists.append(np.sort(r.boxes.conf.cpu().numpy())[::-1])
    grid = np.round(np.arange(0.1, 0.85, 0.025), 3)

    def mae(c):
        pc = np.array([(s >= c).sum() for s in score_lists])
        return float(np.abs(pc - gts).mean())

    best = min(grid, key=mae)
    pc = np.array([(s >= best).sum() for s in score_lists])
    nz = gts > 0
    return {"best_conf": round(float(best), 3), "count_MAE": round(mae(best), 3),
            "count_MAPE": round(float((np.abs(pc - gts)[nz] / gts[nz]).mean() * 100), 2),
            "bias": int((pc - gts).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval")
    args = ap.parse_args()

    if args.eval:
        import json
        r = count_mae(args.eval)
        print(json.dumps(r, indent=2))
        pathlib.Path("compare/followup_h2.json").write_text(json.dumps(
            {"checkpoint": args.eval, "lam": args.lam, **r}, indent=2))
        return

    _STATE["warmup"] = args.warmup
    patch_loss(args.lam)
    from ultralytics import YOLO
    model = YOLO("yolov8s.pt")
    model.add_callback("on_train_epoch_start",
                       lambda tr: _STATE.__setitem__("epoch", tr.epoch))
    model.train(
        data="dataset_official/dataset.yaml", epochs=args.epochs, imgsz=1280, batch=4,
        device="0", workers=0, patience=50, cos_lr=True, seed=args.seed, deterministic=True,
        scale=0.2, close_mosaic=20, mixup=0.0, copy_paste=0.1,
        translate=0.1, degrees=15.0, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, box=8.5, cls=0.3, dfl=1.5,
        project="runs", name=f"yolo_h2_s{args.seed}",
    )


if __name__ == "__main__":
    main()
