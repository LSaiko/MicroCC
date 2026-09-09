# MicroCC — Microscopy Cell Counter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LSaiko/MicroCC/blob/main/bbbc039/usage_example.ipynb)

YOLOv8 nucleus **detection** for counting cells in fluorescence microscopy images.
Counting by detection (not segmentation): each nucleus gets a bounding box, the
count is `len(boxes)`. Faster at inference than instance segmentation and more
robust when nuclei touch.

Trained and evaluated on [BBBC039](https://bbbc.broadinstitute.org/BBBC039)
(U2OS cell nuclei, Hoechst stain, 520×696 single-channel TIFF).

📄 **[REPORT.md](REPORT.md)** — technical report: 7 models benchmarked, and why
the architecture choice mattered less than the detection caps, the label
quality, and the confidence threshold.

**Topics:** `object-detection` · `yolov8` · `cell-counting` · `microscopy` ·
`fluorescence` · `bioimage-analysis` · `ultralytics` · `bbbc039`

## Results

`yolov8s`, 1280 px, evaluated on the 40-image val split at conf 0.4:

| Metric | Value |
|---|---|
| mAP@50 | **0.980** |
| mAP@50:95 | **0.821** |
| Count MAE | **1.8** nuclei/image |
| Count MAPE | **2.1 %** |
| Count bias (pred − gt) | −5 over 3853 |

Baseline `yolov8s` at 640 px / default aug reached only mAP@50 0.55 — see
[Tuning](#tuning) for what moved it. (Count MAE improved from 3.2 → 1.8 when the
GT boxes were rebuilt with watershed instances — see the study below.)

Predicted boxes (left) vs. ground truth (right) on a val batch:

<p float="left">
  <img src="showcase/val_predictions_1.jpg" width="49%" alt="validation predictions" />
  <img src="showcase/val_groundtruth_1.jpg" width="49%" alt="ground truth" />
</p>

More predictions: [showcase/val_predictions_2.jpg](showcase/val_predictions_2.jpg)

### vs. other detectors

RT-DETR-L, RetinaNet, FCOS, Faster R-CNN and a fine-tuned Cellpose, trained on
the same split, one harness. **F1@0.5** = precision/recall at IoU 0.5 (optimal
matching) at each model's *tuned confidence* — the deployed operating point, and
the one number comparable across detection and segmentation.

| Model | F1@0.5 | mAP@50 | best conf | count MAE (tuned) | ms/img |
|---|---|---|---|---|---|
| RT-DETR-L | 0.898 | 0.960 | 0.70 | **1.52** | 51 |
| **YOLOv8s** | 0.899 | 0.975 | 0.40 | 1.82 | **33** |
| Faster R-CNN | 0.899 | 0.962 | 0.70 | 1.95 | 86 |
| Cellpose (fine-tuned) | **0.903** | 0.90¹ | — | 2.27 | 240 |
| FCOS | 0.895 | **0.977** | 0.50 | 2.38 | 67 |
| RetinaNet | 0.889 | 0.950 | 0.50 | 2.58 | 65 |
| Cellpose (zero-shot) | 0.781 | 0.67¹ | — | 16.1 | 452 |

¹ Cellpose has no per-object score — its single operating point, not comparable
to the detectors' swept mAP.

![model comparison](showcase/model_comparison.png)

#### Significance

- **Architecture is not the lever — in-domain.** Six trained models — two-stage,
  anchor-based, anchor-free FCN, YOLO, transformer, and fine-tuned segmentation —
  span F1@0.5 0.889–0.903 and count MAE 1.5–2.6. A 3-seed run on the **official
  test split** ([REPORT.md §2.1](REPORT.md)) confirms it: F1@0.5 0.937–0.943,
  per-model σ ≤ 0.004; a faint Faster R-CNN ≈ RT-DETR ≳ YOLOv8s ordering (~0.6 pt)
  that reverses the mAP@50 order. Re-training the torchvision models with a
  YOLO-equivalent recipe (strong aug, +50 % epochs) doesn't change it — Faster
  R-CNN stays at F1 0.943 ([§2.1 follow-up G](REPORT.md)) — so it isn't an
  under-tuning artifact.
- **…but architecture *is* the lever for robustness.** Zero-shot on DSB2018
  ([§2.9](REPORT.md)), RT-DETR-L holds F1@0.5 **0.80** on out-of-domain
  fluorescence where YOLOv8s drops to **0.67**. The counter transfers within the
  fluorescence-nucleus modality (7–9% count error on a similar assay), degrades
  on far fluorescence (37–43%), and breaks on H&E / brightfield.
- **The per-image detection cap decides it.** torchvision defaults
  (`detections_per_img` 100–300, `topk_candidates` 1000) are set for COCO's ~7
  objects/image and clip recall on fields of 100–165 nuclei. Raising them lifts
  Faster R-CNN mAP@50 0.84 → 0.96 and FCOS 0.84 → 0.98 on the *same weights*.
  **Audit every `*_per_img` / `*_top_n` / `max_det` before benchmarking a dense
  task.**
- **mAP@50 doesn't rank the counting.** FCOS tops mAP@50 (0.977) with a middling
  count MAE (2.38); RT-DETR is 5th on mAP@50, 1st on count MAE (1.52). Read the
  count column.
- **Confidence calibration is per-model** (optimal 0.40–0.70). Hard-coding
  `conf=0.5` and swapping detectors counts wrong.
- **"NMS-free" is not an advantage here** ([§2.8](REPORT.md)). NMS suppresses
  0.0–0.3 % of correct detections on BBBC039; strict NMS (IoU 0.30) is optimal;
  NMS-free RT-DETR-L sits *behind* RetinaNet and FCOS in-domain. RT-DETR's earlier
  "recall lead" was entirely the detection-cap confound.
- **Zero-shot ≠ the ceiling.** Off-the-shelf Cellpose (F1 0.78, count MAE 16)
  looked like a paradigm loss; fine-tuned on the same 158 images it posts the
  **best F1@0.5 of any model (0.903)**. "Detection beats segmentation" was an
  out-of-domain artifact.
- **The F1 ceiling is ~1/3 label, ~2/3 model** ([§2.4](REPORT.md)). Two competent
  labellers disagree on 5–6 % of nuclei; the model tops out at ~0.95–0.96 even
  against the friendliest GT.
- **The counting error is mostly a threshold problem, not a detection problem**
  ([§2.6](REPORT.md)) — a per-image oracle threshold cuts count MAE 2.1 → 0.4.
  But **three attempts to realise that headroom all failed**: a post-hoc learned
  threshold, a threshold-free density-map counter (test MAE 1.94, ~on par with
  the detector), and a count-consistency loss inside YOLO training (degrades
  mAP). Detect-then-**global**-threshold is a hard baseline to beat.
- **Three corrections, each bigger than swapping models:** the detection cap
  (correction 1), the resolution assumption (correction 2, ~+0.01 only), and the
  mask→box GT — rebuilt with watershed instances + a size filter, which lifted
  the F1 ceiling ~0.88 → ~0.90 and dropped count MAE ~40 % across the board
  (YOLO 3.2 → 1.8, RT-DETR 2.5 → 1.5). See [compare/RESULTS.md](compare/RESULTS.md).

#### Which model to use

| If you need… | Pick | Caveat |
|---|---|---|
| Lowest latency, single-package workflow | **YOLOv8s** (33 ms) | conf ≈ 0.40 for counting |
| Best count MAE among detectors | **RT-DETR-L** (1.52) | 960 px / 7 GB VRAM, conf ≈ 0.70 |
| Lowest count error overall + per-nucleus masks | **fine-tuned Cellpose** (F1 0.903, MAE 2.27) | 4–7× slower (240 ms); trains on instance-label masks |
| A `torchvision`-only stack | **Faster R-CNN** or **FCOS** | competitive *only* with `detections_per_img` / `topk_candidates` raised above your max object count |

#### Capacity vs. hardware (measured on RTX 5060, 8 GB)

| Model | Params | Train config that fits 8 GB | Peak VRAM (train) | Infer ms/img¹ | Train time |
|---|---|---|---|---|---|
| YOLOv8s | 11 M | 1280 px, batch 4 | 5.4 GB | 33 | ~30 min (best @ ep 74) |
| RT-DETR-L | 32 M | **960 px**, batch 4 (1280 OOMs) | 7.1 GB | 51 | ~30 min (best @ ep 48) |
| RetinaNet | 36 M | 800 px, batch 2 | ~3 GB | 65 | ~25 min (40 ep) |
| FCOS | 32 M | 800 px, batch 2 | ~3 GB | 67 | ~30 min (40 ep) |
| Faster R-CNN | 43 M | 800 px, batch 2 | ~3 GB | 86 | ~25 min (40 ep) |

¹ YOLO's figure includes image load from disk; the torchvision models were
handed pre-loaded tensors, so YOLO's real inference margin over them is larger.
torchvision VRAM is a fwd+bwd smoke-test estimate; the P2 (stride-4) YOLO
variant OOMs at ≥640 px on 8 GB.

Practical reading: **VRAM is not the constraint** for the torchvision models —
they'd fit a 4 GB card at 1280 px / batch 2. YOLO (5.4 GB) and RT-DETR (7.1 GB)
are the memory-hungry ones. On **≥16 GB**, RT-DETR at 1280 px + larger batch is
the obvious next experiment. CPU-only inference is viable for one-off counts
(~1–3 s/image) but not batches.

#### Proposed future studies

1. ~~**Fair-resolution rematch.**~~ ✅ Resolution was not the lever — the
   per-image detection cap was.
2. ~~**Detection vs. segmentation baseline.**~~ ✅ Fine-tuned Cellpose ties/leads
   the detectors (F1 0.903); zero-shot loses.
3. ~~**Improve the mask→box GT.**~~ ✅ Watershed instances + size filter +
   Hungarian matching. F1 ceiling ~0.88 → ~0.90; count MAE dropped ~40 %.
4. ~~**3-seed run on the official test split (follow-up A).**~~ ✅ Convergence
   confirmed; faint FRCNN ≈ RT-DETR ≳ YOLO ordering.
5. ~~**Cross-dataset generalisation (follow-up B).**~~ ✅ Transfers within
   fluorescence; RT-DETR far more robust than YOLO out-of-domain; breaks on
   H&E/brightfield.
6. ~~**Controlled NMS-free test (follow-up C).**~~ ✅ No NMS-free advantage on
   BBBC039; the earlier recall lead was the detection-cap confound.
7. ~~**Label-vs-model residual (follow-up D).**~~ ✅ Independent labellers disagree
   on 5–6 % of nuclei; ~1/3 of the F1 residual is label ambiguity, ~2/3 model.
8. ~~**Learned per-image threshold (follow-up E).**~~ ✅ Oracle shows ~6× count-MAE
   headroom, but a post-hoc threshold regressor fails.
9. ~~**Count-native density model (follow-up H).**~~ ✅ Test MAE 1.94, on par with
   the detector, still far from the oracle.
10. ~~**Count-consistency loss inside YOLO training (follow-up H2).**~~ ✅ A
    peak-count auxiliary loss degrades mAP without helping the count.
11. ~~**Equal-effort torchvision retrain (follow-up G).**~~ ✅ Strong aug + longer
    schedule leaves Faster R-CNN's F1@0.5 at 0.943 (RetinaNet slightly worse) —
    the architecture convergence is not an under-tuning artifact.
12. **Better-engineered count loss** — Hungarian-matched, or on a DETR-style set
    predictor (research-scale; the oracle headroom is still open).
13. **RT-DETR at 1280 + longer schedule** on a ≥16 GB GPU.
14. **Tiling (SAHI)** — 512 px tiles, 20 % overlap.
15. **StarDist** as a second segmentation baseline.

Full study, corrections, and per-model detail: [compare/RESULTS.md](compare/RESULTS.md).

![training curves](showcase/training_curves.png)
![precision-recall](showcase/pr_curve.png)

## Setup

```bash
pip install ultralytics opencv-python-headless numpy pyyaml scipy scikit-image
```
(`scipy` + `scikit-image` are for the watershed GT construction.)

Download the dataset from https://bbbc.broadinstitute.org/BBBC039 (images +
masks + metadata) and unzip so you have a folder of `*.tif` images and a folder
of `*.png` masks.

## Pipeline

```bash
# 1. masks -> YOLO bbox labels (watershed instances from BBBC039's interior/boundary
#    semantic masks; also handles instance-labelled and binary masks; empty -> empty .txt)
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
- **Counting**: sweep conf on val — MAE minimises near **conf 0.4** with the
  watershed GT (bias −5 over 3853). `evaluate.py` defaults to 0.5; pass `--conf 0.4`.

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
