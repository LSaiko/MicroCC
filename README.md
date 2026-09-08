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
through one torchmetrics harness. **All per-image detection caps raised** (see
significance) — the torchvision models were re-trained at 1280 px too:

| Model | mAP@50 | mAP@75 | mAR@500 | count MAE (tuned) | ms/img |
|---|---|---|---|---|---|
| Faster R-CNN @1280 | **0.976** | 0.955 | 0.900 | 3.0 | 106 |
| FCOS @1280 | **0.976** | **0.959** | **0.927** | 3.2 | 86 |
| **YOLOv8s** @1280 | 0.969 | 0.931 | 0.786 | 3.2 | **34** |
| Faster R-CNN @800 | 0.968 | 0.955 | 0.903 | 2.9 | 89 |
| RT-DETR-L @960 | 0.967 | 0.935 | 0.910 | 2.5 | 56 |
| RetinaNet @800 | 0.956 | 0.935 | 0.891 | 2.6 | 65 |
| RetinaNet @1280 | 0.955 | 0.933 | 0.888 | 3.3 | 80 |
| *Cellpose `nuclei`, zero-shot* | *0.84¹* | — | — | *12.6* | *493* |

¹ Cellpose has no per-object score, so this is its single operating point (F1@0.5
= 0.87), not comparable to the detectors' swept mAP. With **COCO-default caps**
the torchvision checkpoints score Faster R-CNN 0.839 / FCOS 0.838 / RetinaNet
0.881 — what earlier versions of this README reported before the rematch exposed
the cause.

![model comparison](showcase/model_comparison.png)

#### Significance

- **The per-image detection cap is the dominant lever — not architecture, not
  resolution.** torchvision detectors default to `detections_per_img` 100–300 and
  `topk_candidates` 1000, values set for COCO (≈7 objects/image). On fields of
  100–165 nuclei they silently clip recall. Raising them (three config lines)
  lifts Faster R-CNN 0.839 → 0.968 and FCOS 0.838 → 0.975 on the *same weights,
  same resolution* — a +0.13 swing. **Audit every `*_per_img` / `*_top_n` /
  `max_det` before comparing models on a dense-detection task.**
- **With caps raised, architecture barely matters here.** Two-stage,
  anchor-based, anchor-free FCN, anchor-free YOLO, and a transformer all land in
  mAP@50 0.955–0.976 — a 0.02 spread, within noise on a 40-image val set.
- **Resolution 800 → 1280 is minor** (~+0.01 mAP@50) but does tighten FCOS's
  boxes enough to halve its count MAE (5.1 → 3.2).
- **mAP@50 still doesn't rank the counting.** FCOS @800 is 3rd on mAP, worst on
  tuned count MAE; RetinaNet @800 is near-last on mAP, ties for best count MAE.
  Read the count column for a counting deployment.
- **Confidence calibration is per-model** — optimal counting conf 0.45–0.80
  across models. A pipeline that hard-codes `conf=0.5` from a YOLO tutorial and
  swaps in another detector counts wrong.
- **YOLOv8s now has the *lowest* recall of the five** (mAR 0.786). Its edge is
  latency (34 ms, 1.6–3× faster) and a one-package workflow, not accuracy.
- **Zero-shot Cellpose (the domain-standard segmentation tool) loses badly** —
  F1 0.87, count MAE 12.6, +12 % systematic over-count, 493 ms/image. A detector
  trained 15–25 min on 160 labelled images is ~4× more accurate at counting.
  Cellpose wins only when you have *no* labels or need per-nucleus masks.

#### Which model to use

| If you need… | Pick | Caveat |
|---|---|---|
| Lowest latency, single-package workflow | **YOLOv8s** | lowest recall of the five; fine for well-separated nuclei, weakest on confluent fields |
| Highest recall on crowded/confluent fields | **FCOS @1280** or **RT-DETR-L** | 2.5–3× YOLO's latency; FCOS needs its detection caps raised, RT-DETR needs 960 px / 7 GB |
| Best tuned count MAE | **RT-DETR-L** (2.5) / **RetinaNet @800** (2.6) | both need per-model conf (0.70 / 0.45) |
| A `torchvision`-only stack | **Faster R-CNN** or **FCOS** | competitive *only* with `detections_per_img`/`topk_candidates` raised well above your max object count |

#### Capacity vs. hardware (measured on RTX 5060, 8 GB)

| Model | Params | Train config that fits 8 GB | Peak VRAM (train) | Infer ms/img¹ | Train time |
|---|---|---|---|---|---|
| YOLOv8s | 11 M | 1280 px, batch 4 | 5.4 GB (measured) | 34 | ~15 min (best @ ep 43) |
| RT-DETR-L | 32 M | **960 px**, batch 4 (1280 OOMs) | 7.1 GB (measured) | 56 | ~20 min (converged ep 47) |
| RetinaNet | 36 M | 1280 px, batch 2 | ~3 GB | 65–80 | ~25 min (40 ep @800) |
| FCOS | 32 M | 1280 px, batch 2 | ~3 GB | 69–86 | ~100 min @800, ~2× @1280 (slow head) |
| Faster R-CNN | 43 M | 1280 px, batch 2 | ~3 GB | 89–106 | ~25 min (40 ep @800) |

¹ range is @800 → @1280. YOLO's figure includes image load from disk; the
torchvision models were handed pre-loaded tensors, so YOLO's real inference
margin over them is larger. torchvision VRAM is a fwd+bwd smoke-test estimate
(batch-1 peaked ~2.2 GB at 1280 px); the P2 (stride-4) YOLO variant OOMs at
≥640 px on 8 GB.

Practical reading: **VRAM is not the constraint** for the torchvision models —
they'd fit a 4 GB card at 1280 px / batch 2. YOLO (5.4 GB) and RT-DETR (7.1 GB)
are the memory-hungry ones. On **≥16 GB**, RT-DETR at 1280 px + larger batch is
the obvious next experiment. CPU-only inference is viable for one-off counts
(~1–3 s/image) but not batches.

#### Proposed future studies

1. ~~**Fair-resolution rematch.**~~ ✅ Done — see the correction above. Resolution
   was *not* the lever; the per-image detection cap was.
2. **RT-DETR at 1280 + longer schedule** on a ≥16 GB GPU — does it pull clear of
   the pack, or is the 0.97 plateau real?
3. ~~**Detection vs. segmentation baseline.**~~ ✅ Done — zero-shot Cellpose
   `nuclei` scores F1@0.5 0.87 / count MAE 12.6, ~4× worse than the trained
   detectors. Open follow-up: a Cellpose model *fine-tuned* on BBBC039 (the fair
   trained-vs-trained fight); add StarDist.
4. **Tiling (SAHI)** — 512 px tiles at 20 % overlap; may lift recall further on
   the densest fields without a resolution increase.
5. **Cross-dataset generalisation.** Train on BBBC039, evaluate zero-shot on
   BBBC038 / DSB2018 — does the 0.97 cluster survive a domain shift?
6. **Count-calibrated training.** A count-consistency loss or learned per-image
   threshold, so the model optimises the deployed metric directly instead of a
   post-hoc conf sweep.

Full study, correction, and per-model detail: [compare/RESULTS.md](compare/RESULTS.md).

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
