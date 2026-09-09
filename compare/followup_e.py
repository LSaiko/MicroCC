"""Follow-up E — can a learned per-image confidence threshold beat the global one,
i.e. remove the post-hoc conf sweep?

    python compare/followup_e.py

For each image the detector's count is a step function of the confidence
threshold. §2.6 tunes ONE global threshold on val. Here we fit a small regressor
(val -> per-image optimal threshold) from cheap features of the image and the
prediction-score distribution, and test on the official test split:

  global   : one threshold from val (the §2.6 approach)
  learned  : per-image threshold predicted by the regressor
  oracle   : per-image threshold that exactly hits the GT count (lower bound)

Answers REPORT.md §2.6.
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset import YoloDetectionDataset  # noqa: E402

GRID = np.round(np.arange(0.05, 0.90, 0.025), 3)
WEIGHTS = "runs/detect/runs/yolo_s{}/weights/best.pt"
SEEDS = (0, 1, 2)


def collect(ds, weights):
    from ultralytics import YOLO
    m = YOLO(weights)
    feats, oracle, gt_counts, score_lists = [], [], [], []
    for i in range(len(ds)):
        img, target = ds[i]
        gt = len(target["boxes"])
        r = m.predict(str(ds.imgs[i]), conf=0.03, iou=0.6, imgsz=1280,
                      max_det=2000, verbose=False)[0]
        s = np.sort(r.boxes.conf.cpu().numpy())[::-1]
        score_lists.append(s)
        gt_counts.append(gt)
        n = lambda c: int((s >= c).sum())  # noqa: E731
        # features: prediction-count profile + score-distribution shape + image stats
        gimg = img.numpy().mean(0)
        f = [n(0.2), n(0.3), n(0.4), n(0.5), n(0.6),
             n(0.3) / max(n(0.15), 1),
             float(s[:50].mean()) if len(s) else 0.0,
             float(s[:50].std()) if len(s) > 1 else 0.0,
             float(np.median(s)) if len(s) else 0.0,
             float(gimg.mean()), float(gimg.std())]
        feats.append(f)
        # oracle threshold: grid point whose count is closest to GT
        oracle.append(GRID[np.argmin([abs(n(c) - gt) for c in GRID])])
    return np.array(feats), np.array(oracle), np.array(gt_counts), score_lists


def mae_at(score_lists, gt, thr):
    thr = np.broadcast_to(thr, (len(score_lists),))
    pc = np.array([(s >= t).sum() for s, t in zip(score_lists, thr)])
    return float(np.abs(pc - gt).mean())


def run_seed(seed):
    from sklearn.ensemble import GradientBoostingRegressor
    va = YoloDetectionDataset("dataset_official", "val", train=False)
    te = YoloDetectionDataset("dataset_official", "test", train=False)
    Xv, yv, gv, sv = collect(va, WEIGHTS.format(seed))
    Xt, yt, gt, stl = collect(te, WEIGHTS.format(seed))

    # global threshold from val
    g_best = GRID[np.argmin([mae_at(sv, gv, c) for c in GRID])]

    # learned per-image threshold
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                    subsample=0.8, random_state=0)
    reg.fit(Xv, yv)
    pred_thr = np.clip(reg.predict(Xt), GRID.min(), GRID.max())

    return {
        "global_conf": round(float(g_best), 3),
        "MAE_global": round(mae_at(stl, gt, g_best), 3),
        "MAE_learned": round(mae_at(stl, gt, pred_thr), 3),
        "MAE_oracle": round(mae_at(stl, gt, yt), 3),
        "MAPE_global": round(float(np.mean([abs((s >= g_best).sum() - c) / c
                                            for s, c in zip(stl, gt) if c]) * 100), 2),
        "MAPE_learned": round(float(np.mean([abs((s >= t).sum() - c) / c
                                             for s, t, c in zip(stl, pred_thr, gt) if c]) * 100), 2),
    }


def main():
    per_seed = [run_seed(s) for s in SEEDS]
    keys = per_seed[0].keys()
    agg = {k: {"mean": round(float(np.mean([d[k] for d in per_seed])), 3),
               "std": round(float(np.std([d[k] for d in per_seed])), 3)} for k in keys}
    out = {"YOLOv8s": {"per_seed": per_seed, "agg": agg}}
    pathlib.Path("compare/followup_e.json").write_text(json.dumps(out, indent=2))

    a = agg
    print("\nYOLOv8s, official test split, 3 seeds — count MAE:")
    print(f"  global threshold (§2.6 approach) : {a['MAE_global']['mean']:.2f} ± {a['MAE_global']['std']:.2f}"
          f"   (MAPE {a['MAPE_global']['mean']:.1f}%)")
    print(f"  learned per-image threshold      : {a['MAE_learned']['mean']:.2f} ± {a['MAE_learned']['std']:.2f}"
          f"   (MAPE {a['MAPE_learned']['mean']:.1f}%)")
    print(f"  oracle per-image threshold (LB)  : {a['MAE_oracle']['mean']:.2f} ± {a['MAE_oracle']['std']:.2f}")
    print("\nsaved -> compare/followup_e.json")


if __name__ == "__main__":
    main()
