# Comparison study — 5 detectors + Cellpose on BBBC039 nucleus counting

**Question:** does the model choice matter for counting nuclei, or does careful
setup matter more?

**Answer:** setup matters more. With per-image detection caps raised and each
model's confidence tuned, **every trained model lands at F1@0.5 0.89–0.90 and
count MAE 1.5–2.6** — a two-stage CNN, three one-stage CNNs, a transformer, and
a fine-tuned segmentation model, all within noise of each other. They separate
on **speed** and on whether you need instance masks, not on detection quality.
Getting there took three corrections (see *How the conclusions evolved*).

## Models

| Model | Type | Backbone | Params | Source |
|---|---|---|---|---|
| YOLOv8s | one-stage, anchor-free | CSPDarknet | 11 M | `ultralytics` |
| RT-DETR-L | transformer, NMS-free set prediction | HGNetv2 | 32 M | `ultralytics` |
| RetinaNet | one-stage, dense, focal loss | ResNet50-FPN v2 | 36 M | `torchvision` |
| FCOS | one-stage, anchor-free FCN | ResNet50-FPN | 32 M | `torchvision` |
| Faster R-CNN | two-stage (RPN + RoI head) | ResNet50-FPN v2 | 43 M | `torchvision` |
| Cellpose | instance segmentation (U-Net + flows) | — | ~6 M | `cellpose` 3.1 |

All detectors COCO-pretrained, head reset to 1 class, same 160/40 split, hflip
aug. Cellpose run zero-shot (`nuclei`) and fine-tuned on the same 158 train
images (150 ep, ~9 min). YOLO 1280 px, RT-DETR 960 px, torchvision ~800 px.

## Method

- **One harness** (`eval_all.py`): torchmetrics mAP + a per-model confidence
  sweep for count MAE + **F1@0.5** (precision/recall at IoU 0.5, optimal
  Hungarian matching) evaluated **at each model's tuned confidence** — the
  operating point you deploy, and the one number comparable to Cellpose (which
  has no PR curve to sweep).
- **All per-image detection caps raised** at eval (`detections_per_img`,
  `topk_candidates`, `rpn_post_nms_top_n`) — see correction 1.
- **GT boxes** built by watershedding BBBC039's semantic masks from interior
  seeds over the full foreground, with a real size filter — see correction 3.

## Results (40-image val, sorted by count MAE)

| Model | F1@0.5¹ | mAP@50 | mAR@500 | best conf | **count MAE** | count MAPE | ms/img |
|---|---|---|---|---|---|---|---|
| RT-DETR-L | 0.898 | 0.960 | 0.900 | 0.70 | **1.52** | 1.7 % | 51 |
| **YOLOv8s** | 0.899 | 0.975 | 0.834 | 0.40 | 1.82 | 2.1 % | **33** |
| Faster R-CNN | 0.899 | 0.962 | 0.868 | 0.70 | 1.95 | 2.0 % | 86 |
| Cellpose (fine-tuned) | **0.903** | 0.90² | 0.85² | — | 2.27 | 2.1 % | 240 |
| FCOS | 0.895 | **0.977** | **0.911** | 0.50 | 2.38 | 2.5 % | 67 |
| RetinaNet | 0.889 | 0.950 | 0.874 | 0.50 | 2.58 | 2.7 % | 65 |
| Cellpose (zero-shot) | 0.781 | 0.67² | 0.78² | — | 16.1 | 15.4 % | 452 |

¹ optimal 1-to-1 match, IoU ≥ 0.5, at the tuned confidence.
² Cellpose score-1.0 operating point — not comparable to the detectors' swept mAP.

![comparison](comparison.png)

## Findings

1. **Architecture is not the lever.** Six trained models (5 detector families +
   fine-tuned Cellpose) span F1@0.5 0.889–0.903 and count MAE 1.5–2.6. On "find
   every one of many near-identical blobs," the head design does not decide the
   outcome once the model is configured to emit enough detections.

2. **The per-image detection cap decides it.** torchvision's COCO defaults
   (`detections_per_img` 100–300, `topk_candidates` 1000) clip output on fields
   of 100–165 nuclei. Raising them lifts Faster R-CNN from mAP@50 0.84 → 0.96 and
   FCOS 0.84 → 0.98 on the *same weights*. **Audit every `*_per_img` / `*_top_n` /
   `max_det` before comparing models on a dense task.**

3. **Input resolution is a minor lever** (~+0.01 mAP@50 for 800 → 1280 px on the
   torchvision models). The first study wrongly attributed the whole gap to it.

4. **mAP@50 does not rank the counting.** FCOS has the top mAP@50 (0.977) and a
   middling count MAE (2.38); RT-DETR is 5th on mAP@50 and 1st on count MAE
   (1.52). For a counting deployment, read the count column.

5. **Confidence calibration is per-model** — optimal counting conf 0.40–0.70
   across models. Hard-coding `conf=0.5` from a YOLO tutorial and swapping in
   another detector counts wrong (RT-DETR needs 0.70; zero-shot Cellpose's
   scores can't be thresholded at all).

6. **Zero-shot ≠ the method's ceiling.** Off-the-shelf Cellpose (F1 0.78, count
   MAE 16, +15 % over-count) looked like a paradigm loss. Fine-tuned on the same
   158 images it posts the **best F1@0.5 of any model (0.903)** and count MAE
   2.27. "Detection beats segmentation" was an artifact of testing a pretrained
   model out of domain. The real trade-off: fine-tuned Cellpose gives a
   per-nucleus mask (area/shape/intensity) at 4–7× the latency; a detector gives
   a box at 33–86 ms.

## Significance

- **Benchmark hygiene beats model selection.** Three separate default values —
  the detection cap, the input resolution assumption, and the GT box
  construction — each moved the standings more than swapping architectures did.
  Two of the three made a detector family look unusable when it wasn't.
- **Pick on operational grounds.** Lowest latency and a one-package workflow →
  **YOLOv8s** (33 ms). Best count accuracy among detectors → **RT-DETR-L**
  (needs 960 px / 7 GB, conf 0.70). Lowest count error + free instance masks,
  latency no object → **fine-tuned Cellpose**. The F1 spread is 0.015 — within
  run-to-run noise on 40 val images.
- **The ~0.90 F1 ceiling is now mostly real.** Improving the GT (correction 3)
  lifted it from ~0.88; the residual gap to 1.0 is genuine model error plus the
  imperfect mask→box conversion, not a fixable benchmark bug.

## How the conclusions evolved

| Round | What was concluded | Why it was wrong |
|---|---|---|
| 1–2 | "torchvision CNNs plateau ~13 pts below YOLO/RT-DETR; resolution is the lever." | **Correction 1:** COCO-default per-image detection caps clipped recall on dense fields. Raising them closed the gap on the same weights. |
| rematch | "Retrain torchvision at 1280 px to close the gap." | **Correction 2:** resolution added only ~+0.01; the cap was the real cause. Rematch surfaced it. |
| F1 study | "All models converge at F1@0.5 ≈ 0.88; the ceiling is GT quality." | **Correction 3:** rebuilt the GT (watershed instances, drop <25 px² fragments, Hungarian matching), retrained everything. F1 ceiling rose to ~0.90, count MAE dropped ~40 % across the board (YOLO 3.2 → 1.8, RT-DETR 2.5 → 1.5). The "convergence" conclusion held; the numbers moved. |

## Reproduce

```bash
# GT (watershed instances + size filter) -> dataset
python bbbc039/masks_to_yolo.py --images Img/images --masks mask/masks --out labels
python bbbc039/build_dataset.py --images Img/images --labels labels --out dataset

# detectors
python bbbc039/train.py --epochs 200 --patience 60 --imgsz 1280 --batch 4
python compare/train_rtdetr.py --epochs 100 --imgsz 960
python compare/train_tv.py --model fasterrcnn --epochs 40
python compare/train_tv.py --model retinanet  --epochs 40
python compare/train_tv.py --model fcos       --epochs 40

# Cellpose (pip install "cellpose<4" -- downgrades numpy to 2.0.2)
python compare/cellpose_finetune.py --epochs 150

# score everything + plot
python compare/eval_all.py
python compare/cellpose_baseline.py --label "Cellpose (zero-shot)"
python compare/cellpose_baseline.py --model compare/runs/cellpose_ft/models/bbbc039_ft --label "Cellpose (finetuned)"
python compare/plot_results.py
```

Weights are gitignored. RT-DETR weights land under
`runs/detect/compare/runs/rtdetr/` (ultralytics path quirk); `eval_all.py` globs
for them and excludes `rtdetr`/`compare` from the YOLO glob.
