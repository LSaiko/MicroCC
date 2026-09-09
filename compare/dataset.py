"""YOLO-format dataset -> torchvision detection targets.

Reads dataset/images/<split>/*.png + dataset/labels/<split>/*.txt (class cx cy w h,
normalised) and yields (image_tensor[C,H,W] float 0-1, target dict) where target is
{"boxes": FloatTensor[N,4] xyxy pixels, "labels": Int64Tensor[N] (all 1)}.
"""
import pathlib

import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


def _strong_aug():
    """Follow-up G — heavy augmentation to match YOLO's training effort
    (photometric distort + flips + rotation + scale jitter + IoU crop + zoom-out)."""
    return v2.Compose([
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomZoomOut(fill=0, side_range=(1.0, 2.0), p=0.3),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.RandomApply([v2.RandomRotation(15, expand=False)], p=0.5),
        v2.ScaleJitter(target_size=(520, 696), scale_range=(0.7, 1.3), antialias=True),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_size=3),
    ])


class YoloDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, train=False, aug=None):
        root = pathlib.Path(root)
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.imgs = sorted(p for p in self.img_dir.iterdir()
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not self.imgs:
            raise FileNotFoundError(f"no images in {self.img_dir}")
        self.train = train
        self.aug = _strong_aug() if aug == "strong" else None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = read_image(str(p))                      # uint8 [C,H,W]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        _, h, w = img.shape
        img = F.to_dtype(img, torch.float32, scale=True)

        boxes = []
        lbl = self.lbl_dir / f"{p.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = (float(x) for x in line.split())
                x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
                boxes.append([x0, y0, x1, y1])

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.train and self.aug is not None and len(boxes):
            bb = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w))
            tgt = {"boxes": bb, "labels": torch.ones(len(boxes), dtype=torch.int64)}
            img, tgt = self.aug(img, tgt)
            return img, {"boxes": tgt["boxes"].as_subclass(torch.Tensor).float().reshape(-1, 4),
                         "labels": tgt["labels"].as_subclass(torch.Tensor).long()}

        if self.train and self.aug is None and len(boxes) and torch.rand(1).item() < 0.5:
            img = F.hflip(img)
            boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1., 1, -1, 1]) + torch.tensor([w, 0., w, 0])

        return img, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def collate_fn(batch):
    return tuple(zip(*batch))
