# MicroCC — Microscopy Cell Counter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LSaiko/MicroCC/blob/main/bbbc039/usage_example.ipynb)

YOLOv8 nucleus **detection** for counting cells in fluorescence microscopy images.
Counting by detection (not segmentation): each nucleus gets a bounding box, the
count is `len(boxes)`. Faster at inference than instance segmentation and more
robust when nuclei touch.

Trained and evaluated on [BBBC039](https://bbbc.broadinstitute.org/BBBC039)
(U2OS cell nuclei, Hoechst stain, 520×696 single-channel TIFF).

**Topics:** `object-detection` · `yolov8` · `cell-counting` · `microscopy` ·
`fluorescence` · `bioimage-analysis` · `ultralytics` · `bbbc039`

## Results

`yolov8s`, 1280 px, best checkpoint (epoch 43), evaluated on the 40-image val split at conf 0.5:

| Metric | Value |
|---|---|
| mAP@50 | **0.979** |
| mAP@50:95 | **0.820** |
| Count MAE | **3.2** nuclei/image |
| Count MAPE | **3.2 %** |
| Count bias (pred − gt) | +84 over 3990 |

Baseline `yolov8s` at 640 px / default aug reached only mAP@50 0.55 — see
[Tuning](#tuning) for what moved it.

Predicted boxes (left) vs. ground truth (right) on a val batch:

<p float="left">
  <img src="showcase/val_predictions_1.jpg" width="49%" alt="validation predictions" />
  <img src="showcase/val_groundtruth_1.jpg" width="49%" alt="ground truth" />
</p>

More predictions: [showcase/val_predictions_2.jpg](showcase/val_predictions_2.jpg)

### vs. other detectors

RT-DETR-L, RetinaNet, FCOS and Faster R-CNN trained on the same split, scored
through one torchmetrics harness:

| Model | mAP@50 | mAR@500 | count MAE (tuned) | ms/img |
|---|---|---|---|---|
| **YOLOv8s** | **0.969** | 0.786 | 3.2 | **35** |
| RT-DETR-L | 0.967 | **0.910** | **2.5** | 55 |
| RetinaNet | 0.881 | 0.822 | 3.5 | 65 |
| FCOS | 0.838 | 0.796 | 13.8 | 67 |
| Faster R-CNN | 0.839 | 0.785 | 13.4 | 90 |

![model comparison](showcase/model_comparison.png)

#### Significance (for these 5 models, as configured)

- **Resolution beats architecture.** The ~10-pt mAP gap between the two groups is
  input size (1280/960 vs ~800), not RPN vs. dense vs. transformer. Fix
  resolution/tiling *before* comparing model families.
- **mAP@50 doesn't rank counting accuracy.** FCOS and Faster R-CNN have identical
  mAP yet miss 13–14 nuclei/image; RetinaNet's mAP is mediocre but it counts
  fine once tuned. Read the count-MAE column for this task.
- **Confidence calibration is per-model.** Best counting threshold: 0.70
  RT-DETR / 0.50 YOLO / 0.25 RetinaNet — a 3× spread. Hard-coding `conf=0.5`
  from a YOLO tutorial and swapping in RetinaNet silently under-counts ~18 %.
- **NMS-free helps dense fields.** RT-DETR's mAR@500 0.910 vs. everyone else's
  ~0.80: NMS deletes a box overlapping a higher-scoring one, and confluent
  nuclei overlap, so NMS discards correct detections. RT-DETR's set loss runs no
  NMS. This is the exact failure mode "detection over segmentation" exists to
  dodge.
- **Two-stage doesn't pay off here.** Faster R-CNN is the biggest and slowest and
  worst: its RPN proposal budget spreads thin over 100+ near-identical objects,
  capping recall before stage two runs.

#### Which model to use

| If you need… | Pick | Caveat |
|---|---|---|
| Lowest latency, simplest stack | **YOLOv8s** | counting tied with RT-DETR only *after* conf tuning; lower raw recall |
| Best accuracy / recall / tightest boxes; crowded or confluent fields | **RT-DETR-L** | 1.6× slower (55 ms), 3× the params, needs conf ≈ 0.70, 960 px = 7 GB VRAM |
| Already committed to a `torchvision` pipeline | **RetinaNet** | only if retrained at ≥1024 px; must set conf ≈ 0.25 |
| — | ~~FCOS / Faster R-CNN~~ | dominated on every axis at this resolution; no reason to choose them here |

#### Capacity vs. hardware (measured on RTX 5060, 8 GB)

| Model | Params | Train config that fits 8 GB | Peak VRAM (train) | Infer ms/img¹ | Train time |
|---|---|---|---|---|---|
| YOLOv8s | 11 M | 1280 px, batch 4 | 5.4 GB (measured) | 35 | ~15 min (best @ ep 43) |
| RT-DETR-L | 32 M | **960 px**, batch 4 (1280 OOMs) | 7.1 GB (measured) | 55 | ~20 min (converged ep 47) |
| RetinaNet | 36 M | ~800 px, batch 2 | ~5 GB (est.) | 65 | ~25 min (40 ep) |
| FCOS | 32 M | ~800 px, batch 2 | ~5 GB (est.) | 67 | ~100 min (40 ep, slow head) |
| Faster R-CNN | 43 M | ~800 px, batch 2 | ~5 GB (est.) | 90 | ~25 min (40 ep) |

¹ YOLO's figure includes image load from disk; the torchvision models were handed
pre-loaded tensors, so YOLO's real inference margin over them is larger. The P2
(stride-4) YOLO variant OOMs at ≥640 px on 8 GB — not in this table for that
reason.

Practical reading: on an **8 GB** card only YOLOv8s trains at full 1280 px; RT-DETR
needs 960 and the P2-head YOLO variant does not fit at all. On **≤4 GB**, drop to
640 px / batch 2 (expect the mAP hit from finding 1). On **≥16 GB**, RT-DETR at
1280 px and larger batches is the obvious next experiment (see below). CPU-only
inference is viable for one-off counts (~1–3 s/image at 1280) but not for
batches.

#### Proposed future studies

1. **Fair-resolution rematch.** Retrain Faster R-CNN / RetinaNet / FCOS at
   1280 px (`min_size=1280`) to isolate architecture from input size — finding 1
   predicts they close most of the gap; confirm and quantify.
2. **RT-DETR at 1280 + longer schedule** on a ≥16 GB GPU — does it pull clear of
   YOLO on mAP@50, not just recall?
3. **Detection vs. segmentation baseline.** Add Cellpose / StarDist (the
   bioimaging-standard instance-segmentation tools) and compare count MAE — the
   comparison the task framing implies but this study skipped.
4. **Tiling (SAHI) instead of big inputs** — 512 px tiles at 20 % overlap on the
   torchvision nets; cheaper than 1280 px full-frame, may recover the gap on
   modest hardware.
5. **Cross-dataset generalisation.** Train on BBBC039, evaluate zero-shot on
   BBBC038 / DSB2018 — does RT-DETR's NMS-free recall advantage survive a domain
   shift, or is it overfit to this nucleus density?
6. **Count-calibrated training.** Add a count-consistency loss or a learned
   per-image threshold, so the model optimises the deployed metric directly
   instead of relying on a post-hoc conf sweep.

Full study and per-model detail: [compare/RESULTS.md](compare/RESULTS.md).

![training curves](showcase/training_curves.png)
![precision-recall](showcase/pr_curve.png)

## Setup

```bash
pip install ultralytics opencv-python-headless numpy pyyaml
```

Download the dataset from https://bbbc.broadinstitute.org/BBBC039 (images +
masks + metadata) and unzip so you have a folder of `*.tif` images and a folder
of `*.png` masks.

## Pipeline

```bash
# 1. masks -> YOLO bbox labels (handles BBBC039's interior/boundary semantic masks,
#    instance-labelled masks, and plain binary masks; empty masks -> empty .txt)
python bbbc039/masks_to_yolo.py --images Img/images --masks mask/masks --out labels

# 2. build the YOLO dataset: 3-channel PNGs + train/val split + dataset.yaml
python bbbc039/build_dataset.py --images Img/images --labels labels --out dataset --val-frac 0.2
#    (pass --splits <BBBC039_metadata_dir> to use the official 100/50/50 split)

# 3. train
python bbbc039/train.py --epochs 200 --patience 60 --imgsz 1280 --batch 4

# 4. evaluate: mAP@50, mAP@50:95, count MAE, count MAPE -> table + JSON
python bbbc039/evaluate.py --model runs/detect/*/weights/best.pt --data dataset/dataset.yaml
```

`python bbbc039/test_masks_to_yolo.py` runs the label-conversion self-check.

**Just want to count cells with the trained model?** See
[bbbc039/usage_example.ipynb](bbbc039/usage_example.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LSaiko/MicroCC/blob/main/bbbc039/usage_example.ipynb)
— downloads the release weights and runs inference on one image or a folder.
On Colab, set **Runtime → Change runtime type → GPU** before Run all (CPU works
but imgsz 1280 is slow); the first cell uploads your image.

## Tuning

Small objects (~20×20 px nuclei) on an 8 GB GPU:

- **imgsz 640 → 1280** — the single biggest lever for tiny objects.
- **`scale` 0.5 → 0.2** — default mosaic zoom-out was shrinking nuclei further.
- **`close_mosaic=20`, `mixup=0`** — disable aggressive aug near the end.
- **`hsv_h=0`, low `hsv_s/hsv_v`** — fluorescence is effectively single-channel; hue jitter is noise.
- **`box=8.5`** — weight localisation over classification (only one class).
- The P2 (stride-4) head helps in theory but OOMs at ≥640 px on 8 GB with ~100 objects/image; plain `yolov8s` at 1280 is the affordable equivalent.
- **Counting**: the model over-predicts at conf 0.25 (+16 %). A conf sweep on val minimises MAE near **conf 0.5–0.55** — `evaluate.py` defaults to 0.5.

## Notes

- On Windows, `train.py`/`evaluate.py` set `workers=0`: spawned dataloader
  workers each re-import CUDA and can exhaust the paging file.
- Weights and `runs/` are gitignored — retrain with step 3 (~25 min on an RTX 5060).

## License

Dual-licensed under either of

- MIT ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

The BBBC039 dataset is © the Broad Institute, released for use under the terms
on its [dataset page](https://bbbc.broadinstitute.org/BBBC039); cite Caicedo
et al. and the BBBC (Ljosa et al., Nature Methods, 2012) if you use it.
