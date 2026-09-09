"""Follow-up G — does the "architecture barely matters" convergence (§2.1)
survive when the torchvision models get YOLO-equivalent training effort?

    python compare/followup_g.py

Scores, on the official BBBC039 test split:
  <model>_gbase : light aug (hflip only), 40 ep   -- the study's default recipe
  <model>_G     : strong aug (photometric + flips + rotation + scale jitter +
                  IoU crop + zoom-out) + LR warm-up, 80 ep
against the light-aug 3-seed baseline for Faster R-CNN (follow-up A) and YOLOv8s.
"""
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset import YoloDetectionDataset  # noqa: E402
from eval_all import score, tv_preds  # noqa: E402

DATA, SPLIT = "dataset_official", "test"
REF = {  # light-aug, official split
    "YOLOv8s (light, 3 seeds)": 0.937,
    "Faster R-CNN (light, 3 seeds)": 0.943,
}


def main():
    ds = YoloDetectionDataset(DATA, SPLIT, train=False)
    out = {"reference_F1@0.5": REF, "runs": {}}
    for arch in ("fasterrcnn", "retinanet", "fcos"):
        for suffix, label in ((f"{arch}_gbase", "light-aug"), (f"{arch}_G", "strong-aug")):
            w = f"compare/runs/{suffix}/best.pt"
            if not pathlib.Path(w).exists():
                print(f"  ! {w} missing, skip")
                continue
            print(f"scoring {arch} [{label}] ...", flush=True)
            r = score(tv_preds(w, ds, arch), ds)
            out["runs"][f"{arch} ({label})"] = {
                k: r[k] for k in ("F1@0.5", "mAP50", "mAR500", "best_conf",
                                  "count_MAE@best", "count_MAPE@best_pct")}

    pathlib.Path("compare/followup_g.json").write_text(json.dumps(out, indent=2))
    print(f"\nreference (light aug): " +
          "   ".join(f"{k.split()[0]} {v:.3f}" for k, v in REF.items()))
    print(f"\n{'run':<28} {'F1@0.5':>8} {'mAP@50':>8} {'count MAE':>10} {'conf':>6}")
    for k, v in out["runs"].items():
        print(f"{k:<28} {v['F1@0.5']:>8.3f} {v['mAP50']:>8.3f} "
              f"{v['count_MAE@best']:>10.2f} {v['best_conf']:>6}")
    print("\nsaved -> compare/followup_g.json")


if __name__ == "__main__":
    main()
