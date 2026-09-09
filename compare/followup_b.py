"""Follow-up B — cross-dataset zero-shot generalisation.

    python compare/followup_b.py --dsb /path/to/dsb_stage1_train

Trained on BBBC039 (U2OS, Hoechst, 520x696). Evaluated zero-shot on the
Data Science Bowl 2018 / BBBC038 stage1_train set (670 images, many
modalities), bucketed by modality:

  fluor-far   : bright nuclei on dark bg, NOT 520x696  (true cross-domain)
  fluor-near  : bright nuclei on dark bg, 520x696      (same U2OS/Hoechst assay family)
  histology   : coloured (H&E)                          (out of modality — expect failure)
  brightfield : dark nuclei on light bg                 (out of modality — expect failure)

Answers REPORT.md Sec 3 "single dataset" limitation / to-do B: does the
counter transfer, or is it fit to this stain and nucleus density?
"""
import argparse
import glob
import json
import pathlib
import statistics as st
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from eval_all import pr_f1_at_iou  # noqa: E402

CONF_SWEEP = np.round(np.arange(0.15, 0.75, 0.05), 2)
BBBC039_CONF = 0.4  # what you'd deploy from the BBBC039 study


def fluoro_to_rgb(gray):
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = np.zeros((*norm.shape[:2], 3), np.uint8)
    rgb[..., 1] = norm
    return rgb


def bucket(img):
    if img.ndim == 3 and np.abs(img[..., 0].astype(int) - img[..., 1]).mean() > 10:
        return "histology"
    g = img[..., :3].mean(2) if img.ndim == 3 else img
    if g.mean() / 255 < 0.35:
        return "fluor-near" if img.shape[:2] == (520, 696) else "fluor-far"
    return "brightfield"


def gt_boxes(mask_dir):
    boxes = []
    for m in glob.glob(str(mask_dir / "*.png")):
        im = cv2.imread(m, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 3:
            im = im[..., :3].max(axis=2)
        ys, xs = np.where(im > 0)
        if xs.size:
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def load_dsb(root):
    items = []  # (green_rgb_image_path_placeholder, gt_boxes, bucket) -> we pass arrays
    for d in sorted(pathlib.Path(root).iterdir()):
        ip = next(d.glob("images/*.png"), None)
        if ip is None:
            continue
        img = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        g = img[..., :3].mean(2).astype(np.uint8) if img.ndim == 3 else img
        items.append((fluoro_to_rgb(g), gt_boxes(d / "masks"), bucket(img)))
    return items


def score_model(model_call, items):
    """model_call(rgb) -> (boxes Nx4 xyxy, scores N). Returns per-bucket metrics."""
    by_bucket = {}
    for rgb, gt, bk in items:
        boxes, scores = model_call(rgb)
        by_bucket.setdefault(bk, []).append((boxes, scores, gt))

    out = {}
    for bk, rows in by_bucket.items():
        gts = np.array([len(gt) for _, _, gt in rows])
        nz = gts > 0

        def stats_at(c):
            pc = np.array([(sc >= c).sum() for _, sc, _ in rows])
            ae = np.abs(pc - gts)
            f1s = [pr_f1_at_iou(bx[sc >= c], gt, 0.5) for bx, sc, gt in rows]
            f1 = float(np.mean([f for _, _, f in f1s]))
            return float(ae.mean()), float((ae[nz] / gts[nz]).mean() * 100), f1

        mae_d, mape_d, f1_d = stats_at(BBBC039_CONF)
        best_c = min(CONF_SWEEP, key=lambda c: stats_at(c)[0])
        mae_b, mape_b, f1_b = stats_at(best_c)
        out[bk] = {
            "n_images": len(rows), "n_nuclei": int(gts.sum()),
            "F1@0.5_conf0.4": round(f1_d, 4),
            "count_MAE_conf0.4": round(mae_d, 2), "count_MAPE_conf0.4": round(mape_d, 1),
            "best_conf": round(float(best_c), 2),
            "F1@0.5_best": round(f1_b, 4),
            "count_MAE_best": round(mae_b, 2), "count_MAPE_best": round(mape_b, 1),
        }
    return out


def yolo_caller(weights):
    from ultralytics import YOLO
    m = YOLO(weights)

    def call(rgb):
        r = m.predict(rgb, conf=0.05, iou=0.6, imgsz=1280, max_det=2000, verbose=False)[0].boxes
        return r.xyxy.cpu().numpy(), r.conf.cpu().numpy()
    return call


def rtdetr_caller(weights):
    from ultralytics import RTDETR
    m = RTDETR(weights)

    def call(rgb):
        r = m.predict(rgb, conf=0.05, iou=0.6, imgsz=960, max_det=2000, verbose=False)[0].boxes
        return r.xyxy.cpu().numpy(), r.conf.cpu().numpy()
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsb", required=True, help="dir with the 670 DSB2018 image folders")
    args = ap.parse_args()

    print("loading DSB2018 ...", flush=True)
    items = load_dsb(args.dsb)
    from collections import Counter
    print("buckets:", dict(Counter(bk for _, _, bk in items)), "\n")

    specs = {
        "YOLOv8s": ("yolo", [f"runs/detect/runs/yolo_s{s}/weights/best.pt" for s in (0, 1, 2)]),
        "RT-DETR-L": ("rtdetr", [f"runs/detect/compare/runs/rtdetr_s{s}/weights/best.pt" for s in (0, 1, 2)]),
    }
    results = {}
    for label, (arch, weights_list) in specs.items():
        per_seed = []
        for w in weights_list:
            if not pathlib.Path(w).exists():
                print(f"  ! {label}: {w} missing, skip")
                continue
            print(f"scoring {label} <- {w} ...", flush=True)
            caller = yolo_caller(w) if arch == "yolo" else rtdetr_caller(w)
            per_seed.append(score_model(caller, items))
        if not per_seed:
            continue
        agg = {}
        for bk in per_seed[0]:
            agg[bk] = {"n_images": per_seed[0][bk]["n_images"],
                       "n_nuclei": per_seed[0][bk]["n_nuclei"]}
            for k in ("F1@0.5_conf0.4", "count_MAE_conf0.4", "count_MAPE_conf0.4",
                      "F1@0.5_best", "count_MAE_best", "count_MAPE_best"):
                vals = [s[bk][k] for s in per_seed]
                agg[bk][k] = {"mean": round(st.mean(vals), 3),
                              "std": round(st.pstdev(vals), 3) if len(vals) > 1 else 0.0}
        results[label] = agg

    pathlib.Path("compare/followup_b.json").write_text(json.dumps(results, indent=2))

    for label, agg in results.items():
        print(f"\n=== {label} (zero-shot, {len(specs[label][1])} seeds) ===")
        print(f"{'bucket':<12} {'n_img':>6} {'F1@.5 (conf.4)':>16} {'MAE (conf.4)':>14} "
              f"{'F1@.5 (best)':>14} {'MAE (best)':>12}")
        for bk in ("fluor-far", "fluor-near", "histology", "brightfield"):
            if bk not in agg:
                continue
            a = agg[bk]
            def c(k, f="{:.3f}"):
                return f"{f.format(a[k]['mean'])}±{f.format(a[k]['std'])}"
            print(f"{bk:<12} {a['n_images']:>6} {c('F1@0.5_conf0.4'):>16} "
                  f"{c('count_MAE_conf0.4','{:.1f}'):>14} {c('F1@0.5_best'):>14} "
                  f"{c('count_MAE_best','{:.1f}'):>12}")
    print("\nsaved -> compare/followup_b.json")


if __name__ == "__main__":
    main()
