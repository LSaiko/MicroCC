"""Train a torchvision detector on the BBBC039 cell dataset.

    python compare/train_tv.py --model fasterrcnn --epochs 40
    python compare/train_tv.py --model retinanet  --epochs 40

Same dataset/split as the YOLO model. workers=0 (Windows paging-file limit).
Saves best-by-val-mAP@50 weights to compare/runs/<model>/best.pt.
"""
import argparse
import pathlib
import sys

import torch
from torchmetrics.detection import MeanAveragePrecision

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from dataset import YoloDetectionDataset, collate_fn  # noqa: E402

MODELS = {
    "fasterrcnn": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights"),
    "retinanet": ("retinanet_resnet50_fpn_v2", "RetinaNet_ResNet50_FPN_V2_Weights"),
    "fcos": ("fcos_resnet50_fpn", "FCOS_ResNet50_FPN_Weights"),
}


def build(name, imgsz=None):
    import functools

    import torchvision.models.detection as det
    fn_name, w_name = MODELS[name]
    weights = getattr(det, w_name).COCO_V1
    model = getattr(det, fn_name)(weights=weights, num_classes=91)
    if imgsz:  # match a target input resolution (default recipe is min_size=800)
        model.transform.min_size = (imgsz,)
        model.transform.max_size = imgsz
    # swap classification head for 2 classes (bg + cell)
    if name == "fasterrcnn":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    elif name == "retinanet":
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead
        n_anchors = model.head.classification_head.num_anchors
        in_ch = model.backbone.out_channels
        model.head.classification_head = RetinaNetClassificationHead(
            in_ch, n_anchors, 2, norm_layer=functools.partial(torch.nn.GroupNorm, 32)
        )
    elif name == "fcos":
        from torchvision.models.detection.fcos import FCOSClassificationHead
        n_anchors = model.head.classification_head.num_anchors
        in_ch = model.backbone.out_channels
        model.head.classification_head = FCOSClassificationHead(in_ch, n_anchors, 2)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        preds = model(imgs)
        preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
        metric.update(preds, [{k: v for k, v in t.items()} for t in targets])
    m = metric.compute()
    return float(m["map_50"]), float(m["map"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--imgsz", type=int, default=0, help="0 = torchvision default (~800)")
    ap.add_argument("--tag", default="", help="output dir suffix, e.g. '1280'")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = pathlib.Path("compare/runs") / (args.model + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    tr = YoloDetectionDataset(args.data, "train", train=True)
    va = YoloDetectionDataset(args.data, "val", train=False)
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                     num_workers=0, collate_fn=collate_fn)
    vl = torch.utils.data.DataLoader(va, batch_size=args.batch, shuffle=False,
                                     num_workers=0, collate_fn=collate_fn)

    model = build(args.model, args.imgsz or None).to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        opt.zero_grad()
        for step, (imgs, targets) in enumerate(tl):
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                losses = model(imgs, targets)
                loss = sum(losses.values()) / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
        sched.step()

        map50, map5095 = evaluate(model, vl, device)
        print(f"epoch {epoch:3d}  loss {running / len(tl):.3f}  "
              f"mAP@50 {map50:.4f}  mAP@50:95 {map5095:.4f}", flush=True)
        if map50 > best:
            best = map50
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "imgsz": args.imgsz, "map50": map50,
                        "map5095": map5095, "epoch": epoch}, out / "best.pt")
    print(f"done. best mAP@50 {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
