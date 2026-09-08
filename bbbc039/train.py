"""Train YOLOv8s on the BBBC039 cell dataset.

    python train.py [--epochs 100] [--imgsz 1280] [--batch 4] [--model yolov8s.pt]

Notes for this box (RTX 5060, 8 GB):
- workers=0 is mandatory (spawned dataloaders blow the Windows paging file).
- The P2 head (yolov8s-p2.yaml) OOMs at >=640 here; plain yolov8s at imgsz 1280
  is the affordable version of the same "see small objects" lever.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/dataset.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=0, patience=args.patience, cos_lr=True, multi_scale=False,
        scale=0.2, close_mosaic=20, mixup=0.0, copy_paste=0.1,
        translate=0.1, degrees=15.0, fliplr=0.5, flipud=0.5,
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
        box=8.5, cls=0.3, dfl=1.5,
        project="runs", name="bbbc039",
    )


if __name__ == "__main__":
    main()
