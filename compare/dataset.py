"""YOLO-format dataset -> torchvision detection targets.

Reads dataset/images/<split>/*.png + dataset/labels/<split>/*.txt (class cx cy w h,
normalised) and yields (image_tensor[C,H,W] float 0-1, target dict) where target is
{"boxes": FloatTensor[N,4] xyxy pixels, "labels": Int64Tensor[N] (all 1)}.
"""
import pathlib

import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class YoloDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, train=False):
        root = pathlib.Path(root)
        self.img_dir = root / "images" / split
        self.lbl_dir = root / "labels" / split
        self.imgs = sorted(p for p in self.img_dir.iterdir()
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not self.imgs:
            raise FileNotFoundError(f"no images in {self.img_dir}")
        self.train = train

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

        if self.train and boxes and torch.rand(1).item() < 0.5:   # hflip
            img = F.hflip(img)
            boxes = [[w - x1, y0, w - x0, y1] for x0, y0, x1, y1 in boxes]

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
