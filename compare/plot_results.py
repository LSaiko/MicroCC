"""compare/results.json -> compare/comparison.png (grouped bars, sorted by mAP@50)."""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

r = json.loads(pathlib.Path("compare/results.json").read_text())
models = sorted(r, key=lambda m: r[m]["count_MAE@best"])
palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(models)))
fig, axes = plt.subplots(1, 3, figsize=(max(13, 1.7 * len(models)), 4.5))

for ax, (key, title, fmt) in zip(axes, [
    ("F1@0.5", "F1 @ IoU 0.5, tuned conf (higher better)", "{:.3f}"),
    ("count_MAE@best", "Count MAE @ best conf (lower better)", "{:.1f}"),
    ("ms_per_image", "Inference ms/image (lower better)", "{:.0f}"),
]):
    vals = [r[m][key] for m in models]
    bars = ax.barh(range(len(models)), vals, color=palette)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    if key == "ms_per_image" and max(vals) / min(vals) > 8:
        ax.set_xscale("log")
    for b, v in zip(bars, vals):
        ax.text(v, b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                va="center", fontsize=8)
    ax.margins(x=0.15)

fig.suptitle("BBBC039 cell counting — 5 detectors + Cellpose, at each model's tuned operating point (40-image val)",
             fontsize=10)
fig.tight_layout()
fig.savefig("compare/comparison.png", dpi=120)
print("wrote compare/comparison.png")
