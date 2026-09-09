"""Follow-up A — score the 3-seed checkpoints on the official BBBC039 test split
and aggregate mean +/- std.

    python compare/followup_a_eval.py

Expects checkpoints from followup_a_train (compare/runs/followup_a/...). Writes
compare/followup_a.json and prints a table. Findings it hardens: REPORT.md
Sec 2.1 (is the F1 convergence real or 40-img noise?) and Sec 2.7.
"""
import glob
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset import YoloDetectionDataset  # noqa: E402
from eval_all import rtdetr_preds, score, tv_preds, yolo_preds  # noqa: E402

DATA = "dataset_official"
RUNS = pathlib.Path("compare/runs/followup_a")

# (label, arch, how to find best.pt for a given seed dir)
SPECS = {
    "YOLOv8s": ("yolo", "yolo_s{seed}"),
    "RT-DETR-L": ("rtdetr", "rtdetr_s{seed}"),
    "Faster R-CNN": ("fasterrcnn", "fasterrcnn_s{seed}"),
}
SEEDS = [0, 1, 2]
KEYS = ["F1@0.5", "mAP50", "mAR500", "count_MAE@best", "best_conf"]


def find_weights(arch, seed):
    if arch == "yolo":
        g = glob.glob(f"runs/detect/**/yolo_s{seed}*/weights/best.pt", recursive=True)
    elif arch == "rtdetr":
        g = glob.glob(f"runs/detect/**/rtdetr_s{seed}*/weights/best.pt", recursive=True)
    else:
        g = glob.glob(f"compare/runs/fasterrcnn_s{seed}/best.pt")
    return sorted(g, key=lambda p: pathlib.Path(p).stat().st_mtime)[-1] if g else None


def preds_for(arch, weights, ds):
    if arch == "yolo":
        return yolo_preds(weights, ds)
    if arch == "rtdetr":
        return rtdetr_preds(weights, ds)
    return tv_preds(weights, ds, "fasterrcnn")


def main():
    test_ds = YoloDetectionDataset(DATA, "test", train=False)
    print(f"test split: {len(test_ds)} images\n")

    out = {}
    for label, (arch, _) in SPECS.items():
        per_seed = []
        for seed in SEEDS:
            w = find_weights(arch, seed)
            if not w:
                print(f"  ! {label} seed {seed}: no checkpoint, skipping")
                continue
            print(f"scoring {label} seed {seed} ({w}) ...", flush=True)
            r = score(preds_for(arch, w, test_ds), test_ds)
            per_seed.append(r)
        if not per_seed:
            continue
        agg = {}
        for k in KEYS:
            vals = [s[k] for s in per_seed if s.get(k) is not None]
            if not vals:
                continue
            agg[k] = {"mean": round(st.mean(vals), 4),
                      "std": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                      "runs": vals}
        out[label] = {"n_seeds": len(per_seed), **agg}

    pathlib.Path("compare/followup_a.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'model':<14} {'F1@0.5':>16} {'mAP@50':>16} {'count MAE':>16}")
    for label, a in out.items():
        def cell(k, fmt="{:.3f}"):
            m = a.get(k)
            return f"{fmt.format(m['mean'])} +/- {fmt.format(m['std'])}" if m else "-"
        print(f"{label:<14} {cell('F1@0.5'):>16} {cell('mAP50'):>16} "
              f"{cell('count_MAE@best','{:.2f}'):>16}")
    print("\nsaved -> compare/followup_a.json")


if __name__ == "__main__":
    main()
