"""compare/results.json -> compare/comparison.png (grouped bars)."""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

r = json.loads(pathlib.Path("compare/results.json").read_text())
models = list(r)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, (key, title, fmt) in zip(axes, [
    ("mAP50", "mAP@50 (higher better)", "{:.3f}"),
    ("count_MAE@best", "Count MAE @ best conf (lower better)", "{:.1f}"),
    ("ms_per_image", "Inference ms/image (lower better)", "{:.0f}"),
]):
    vals = [r[m][key] for m in models]
    bars = ax.bar(models, vals, color=["#2a9d8f", "#e76f51", "#e9c46a"])
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                ha="center", va="bottom", fontsize=9)
    ax.margins(y=0.15)

fig.suptitle("BBBC039 cell detection — YOLOv8s vs torchvision baselines (40-image val)", fontsize=11)
fig.tight_layout()
fig.savefig("compare/comparison.png", dpi=120)
print("wrote compare/comparison.png")
