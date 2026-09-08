"""Follow-up A — train YOLOv8s / RT-DETR-L / Faster R-CNN x 3 seeds on the
official BBBC039 split (100 train / 50 val / 50 test).

    python compare/followup_a_train.py           # all 9 runs, skips existing
    python compare/followup_a_train.py --only yolo

Then: python compare/followup_a_eval.py
"""
import argparse
import glob
import subprocess
import sys

DATA = "dataset_official/dataset.yaml"
SEEDS = [0, 1, 2]

JOBS = {
    "yolo": lambda s: [
        sys.executable, "bbbc039/train.py", "--data", DATA, "--seed", str(s),
        "--epochs", "150", "--patience", "50", "--imgsz", "1280", "--batch", "4",
        "--project", "runs", "--name", f"yolo_s{s}",
    ],
    "rtdetr": lambda s: [
        sys.executable, "compare/train_rtdetr.py", "--data", DATA, "--seed", str(s),
        "--epochs", "80", "--imgsz", "960", "--batch", "4",
        "--project", "compare/runs", "--name", f"rtdetr_s{s}",
    ],
    "fasterrcnn": lambda s: [
        sys.executable, "compare/train_tv.py", "--model", "fasterrcnn", "--data",
        "dataset_official", "--seed", str(s), "--epochs", "40", "--tag", f"s{s}",
    ],
}

DONE_GLOBS = {
    "yolo": "runs/detect/**/yolo_s{s}*/weights/best.pt",
    "rtdetr": "runs/detect/**/rtdetr_s{s}*/weights/best.pt",
    "fasterrcnn": "compare/runs/fasterrcnn_s{s}/best.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(JOBS), action="append")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    which = args.only or list(JOBS)

    for name in which:
        for s in SEEDS:
            if not args.force and glob.glob(DONE_GLOBS[name].format(s=s), recursive=True):
                print(f"=== SKIP {name} seed {s} (checkpoint exists) ===", flush=True)
                continue
            print(f"=== TRAIN {name} seed {s} ===", flush=True)
            rc = subprocess.call(JOBS[name](s))
            print(f"=== {name} seed {s} exit {rc} ===", flush=True)
    print("FOLLOWUP-A-TRAIN-DONE", flush=True)


if __name__ == "__main__":
    main()
