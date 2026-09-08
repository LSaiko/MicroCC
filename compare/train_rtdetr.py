"""Train RT-DETR-L (transformer detector, NMS-free) on the BBBC039 dataset.

    python compare/train_rtdetr.py --epochs 100 --imgsz 960 --batch 4

Same dataset.yaml / split as YOLO. RT-DETR is heavier than yolov8s; imgsz 960
fits the RTX 5060 8 GB, 1280 OOMs. workers=0 (Windows).
"""
import argparse

from ultralytics import RTDETR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="rtdetr")
    ap.add_argument("--project", default="compare/runs")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = RTDETR("rtdetr-l.pt")
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, patience=40, cos_lr=True,
        seed=args.seed, deterministic=True,
        scale=0.2, close_mosaic=20, mixup=0.0,
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
        project=args.project, name=args.name,
    )


if __name__ == "__main__":
    main()
