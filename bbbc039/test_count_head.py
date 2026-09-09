"""Run: python bbbc039/test_count_head.py  (from the repo root)"""
import os
import pathlib

import torch

os.chdir(pathlib.Path(__file__).resolve().parent.parent)  # repo root
from count_head import POOL, SCALE, DensityDataset, build, density_map, dmap

# density map must conserve mass: integral == number of points
pts = [(50, 40), (120, 200), (300, 100), (10, 10)]  # incl. a near-edge point
dm = density_map(pts, 320, 256)
assert abs(dm.sum() - len(pts)) < 1e-4, dm.sum()

# empty -> zero map, no divide-by-zero
assert density_map([], 320, 256).sum() == 0.0

# dataset item: target integrates to the label count, at the pooled resolution
ds = DensityDataset("dataset_official", "val", train=False)
x, d, cnt = ds[0]
assert d.shape[-2] == x.shape[-2] // POOL and d.shape[-1] == x.shape[-1] // POOL
assert abs(float(d.sum()) / SCALE - float(cnt)) < 1e-2, (float(d.sum()) / SCALE, float(cnt))

# dmap() pools the U-Net output to the density grid and stays non-negative
m = build().eval()
with torch.no_grad():
    p = dmap(m, x[None])
assert p.shape == d[None].shape and float(p.min()) >= 0.0

print("ok")
