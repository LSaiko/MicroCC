# Comparison study — YOLOv8s vs. two-stage / dense baselines

Does the YOLO choice actually matter for counting nuclei, or would a standard
detector do as well? Trained two common alternatives on the **same** BBBC039
split and scored all three through **one** metric harness.

## Models

| Model | Type | Backbone | Params | Source |
|---|---|---|---|---|
| **YOLOv8s** | one-stage, anchor-free | CSPDarknet | 11.1 M | `ultralytics` |
| **Faster R-CNN** | two-stage (RPN + RoI head) | ResNet50-FPN v2 | 43.3 M | `torchvision` |
| **RetinaNet** | one-stage, dense, focal loss | ResNet50-FPN v2 | 36.4 M | `torchvision` |

All three: COCO-pretrained, head reset to 1 class, same 160/40 train/val split,
horizontal-flip aug only, trained on this RTX 5060 (8 GB).
YOLO: 1280 px, ~100 epochs (best @43). torchvision: native multi-scale
(~800 px), SGD 5e-3 cosine, 40 epochs (Faster R-CNN best @7, RetinaNet @14 —
both plateaued almost immediately).

## Results (40-image val split, one torchmetrics harness)

| Model | mAP@50 | mAP@75 | mAR@500 | count MAE @conf 0.5 | best conf | count MAE @best | count MAPE @best | ms/img |
|---|---|---|---|---|---|---|---|---|
| **YOLOv8s** | **0.969** | **0.931** | 0.786 | **3.2** | 0.50 | **3.2** | **3.2 %** | **35** |
| RetinaNet | 0.881 | 0.861 | **0.822** | 17.9 | 0.25 | 3.5 | 3.5 % | 63 |
| Faster R-CNN | 0.839 | 0.837 | 0.785 | 14.8 | 0.80 | 13.4 | 11.3 % | 86 |

![comparison](comparison.png)

mAP@50:95 is omitted: torchmetrics caps that metric at 100 detections/image
internally, which unfairly penalises dense predictors on fields of up to ~165
nuclei. mAP@50 / mAP@75 / mAR use a 500-detection cap.

## Findings

1. **YOLOv8s wins on detection quality outright** — mAP@50 0.969 vs 0.881 / 0.839,
   and the gap widens at mAP@75 (tighter boxes). Training at 1280 px is the main
   reason: the torchvision models run at ~800 px where a 20 px nucleus is ~12 px.

2. **Counting accuracy is confidence-threshold-bound, per model.** At a fixed
   conf 0.5, RetinaNet looks terrible (MAE 17.9, misses 18 % of nuclei) — but
   that is calibration, not capability. Focal-loss training pushes its scores
   low; at **conf 0.25** its count MAE drops to **3.5**, matching YOLO. Anyone
   swapping detectors must re-sweep the threshold (as done for YOLO in the main
   README).

3. **Faster R-CNN is dominated on every axis** — worst mAP, worst counting even
   after tuning (MAE 13.4), lowest recall (mAR 0.785), slowest (86 ms). The
   two-stage RPN bottlenecks recall on crowded fields; proposals saturate before
   every nucleus is covered.

4. **Speed:** YOLOv8s 35 ms < RetinaNet 63 ms < Faster R-CNN 86 ms per image
   (RTX 5060). YOLO's number includes image load from disk; the torchvision
   models were handed pre-loaded tensors, so YOLO's real margin is larger.

**Conclusion:** for this task YOLOv8s is the right default — best detection,
best out-of-the-box counting, fastest. RetinaNet is a viable second choice *if*
you tune its confidence threshold and can afford 2× the latency. Faster R-CNN
offers nothing here.

## Reproduce

```bash
python compare/train_tv.py --model fasterrcnn --epochs 40
python compare/train_tv.py --model retinanet  --epochs 40
python compare/eval_all.py        # scores YOLO + both, writes results.json
python compare/plot_results.py    # comparison.png
```

Weights (`compare/runs/*/best.pt`) are gitignored — retrain with the above
(~20 min each on an RTX 5060).
