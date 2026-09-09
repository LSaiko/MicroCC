"""Follow-up H — a count-native model: density-map regression instead of
detection + threshold.

    python bbbc039/count_head.py --data dataset_official --epochs 150
    python bbbc039/count_head.py --data dataset_official --eval runs/count_head/best.pt

A ResNet18-U-Net predicts a density map; the count is its integral, pooled to
stride 8 and scaled so the regression target is O(1). GT density = a unit-mass
Gaussian at each nucleus centroid (from the YOLO-format boxes). No detection,
no NMS, no confidence threshold.

E (REPORT.md §2.6) showed a per-image oracle threshold could cut the detector's
count MAE 2.13 -> 0.36, but a post-hoc threshold regressor couldn't get there.
This tests whether a model trained directly on the count does.
"""
import argparse
import pathlib
import random

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

POOL = 8       # density resolution = input / POOL
SIGMA = 2.0    # Gaussian sigma in *pooled* px  (=16 px at full res)
SCALE = 1000.0  # density target scaled up so MSE is O(1); count = sum / SCALE
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def centroids(label_path, w, h):
    pts = []
    for line in pathlib.Path(label_path).read_text().splitlines():
        if line.strip():
            _, cx, cy, _, _ = (float(x) for x in line.split())
            pts.append((cx * w, cy * h))
    return pts


def density_map(pts, w, h):
    """unit-mass Gaussian per point on the pooled grid; sums to len(pts)."""
    gh, gw = h // POOL, w // POOL
    dm = np.zeros((gh, gw), np.float32)
    for x, y in pts:
        xi, yi = int(x / POOL), int(y / POOL)
        if 0 <= yi < gh and 0 <= xi < gw:
            dm[yi, xi] += 1.0
    dm = gaussian_filter(dm, SIGMA, mode="constant")
    if dm.sum() > 1e-6:                       # conserve mass: integral == count
        dm *= len(pts) / dm.sum()
    return dm


class DensityDataset(torch.utils.data.Dataset):
    def __init__(self, data, split, crop=384, train=True):
        d = pathlib.Path(data)
        self.imgs = sorted((d / "images" / split).glob("*.png"))
        self.lbls = d / "labels" / split
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        g = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        g = g[..., 1] if g.ndim == 3 else g
        h, w = g.shape
        pts = centroids(self.lbls / f"{p.stem}.txt", w, h)

        if self.train:
            c = self.crop
            x0, y0 = random.randint(0, max(0, w - c)), random.randint(0, max(0, h - c))
            g = g[y0:y0 + c, x0:x0 + c]
            pts = [(x - x0, y - y0) for x, y in pts if x0 <= x < x0 + c and y0 <= y < y0 + c]
            gh, gw = g.shape
            if random.random() < 0.5:
                g = g[:, ::-1]; pts = [(gw - 1 - x, y) for x, y in pts]
            if random.random() < 0.5:
                g = g[::-1, :]; pts = [(x, gh - 1 - y) for x, y in pts]
            g = np.ascontiguousarray(g).astype(np.float32) * random.uniform(0.8, 1.2)

        gh, gw = g.shape
        gh, gw = gh - gh % POOL, gw - gw % POOL
        g = g[:gh, :gw]
        pts = [(x, y) for x, y in pts if x < gw and y < gh]

        norm = cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        rgb = (np.stack([norm] * 3, -1) - MEAN) / STD
        dm = density_map(pts, gw, gh) * SCALE
        return (torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
                torch.from_numpy(dm[None]).float(),
                torch.tensor(float(len(pts))))


def build():
    return smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)


def dmap(model, x):
    """model -> pooled, non-negative density (still x SCALE)."""
    return F.avg_pool2d(F.relu(model(x)), POOL) * (POOL * POOL)


def predict_count(model, img_path, device):
    g = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    g = g[..., 1] if g.ndim == 3 else g
    h, w = g.shape
    H, W = h + (-h % 32), w + (-w % 32)
    gp = np.zeros((H, W), np.float32); gp[:h, :w] = g
    norm = cv2.normalize(gp, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    rgb = (np.stack([norm] * 3, -1) - MEAN) / STD
    x = torch.from_numpy(rgb.transpose(2, 0, 1)[None]).float().to(device)
    with torch.no_grad():
        return float(dmap(model, x).sum().item() / SCALE)


def evaluate(model, data, split, device):
    d = pathlib.Path(data)
    imgs = sorted((d / "images" / split).glob("*.png"))
    gts = np.array([len(centroids(d / "labels" / split / f"{p.stem}.txt", 1, 1)) for p in imgs])
    pcs = np.array([predict_count(model, p, device) for p in imgs])
    ae = np.abs(pcs - gts); nz = gts > 0
    return {"MAE": round(float(ae.mean()), 3),
            "MAPE": round(float((ae[nz] / gts[nz]).mean() * 100), 2),
            "bias": round(float((pcs - gts).mean()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_official")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = pathlib.Path("runs/count_head"); out.mkdir(parents=True, exist_ok=True)
    model = build().to(device)

    if args.eval:
        model.load_state_dict(torch.load(args.eval, map_location=device)); model.eval()
        for sp in ("val", "test"):
            print(f"{sp}: {evaluate(model, args.data, sp, device)}")
        return

    tl = torch.utils.data.DataLoader(DensityDataset(args.data, "train", train=True),
                                     batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, dm, cnt in tl:
            x, dm, cnt = x.to(device), dm.to(device), cnt.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = dmap(model, x)
                loss = F.mse_loss(pred, dm) + 0.5 * (pred.sum((1, 2, 3)) / SCALE - cnt).abs().mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        if ep % 5 == 0 or ep == args.epochs:
            model.eval()
            v = evaluate(model, args.data, "val", device)
            print(f"epoch {ep:3d}  loss {tot/len(tl):.3f}  val MAE {v['MAE']:.2f} "
                  f"MAPE {v['MAPE']:.1f}% bias {v['bias']:+.1f}", flush=True)
            if v["MAE"] < best:
                best = v["MAE"]
                torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt")); model.eval()
    print("\nBEST checkpoint:")
    for sp in ("val", "test"):
        print(f"  {sp}: {evaluate(model, args.data, sp, device)}")


if __name__ == "__main__":
    main()
