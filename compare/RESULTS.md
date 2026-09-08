# Comparison study — YOLOv8s vs. 4 detectors + a segmentation baseline

Does the detector choice matter for counting nuclei? Trained four alternatives
on the **same** BBBC039 split, scored all five through **one** metric harness,
ran a fair-resolution rematch that **overturned the first conclusion**, and
added a zero-shot Cellpose segmentation baseline.

> **Correction (this supersedes the earlier version of this file).** Rounds 1–2
> reported Faster R-CNN and FCOS at mAP@50 ~0.84 and concluded "two-stage and
> anchor-free FCN are dominated; only YOLO/RT-DETR are competitive; resolution is
> the lever." That was a **benchmark artifact.** torchvision detectors default to
> a per-image output cap (`detections_per_img` 100–300, `topk_candidates` 1000)
> set for COCO, where images hold ~7 objects. On fields of 100–165 nuclei those
> caps silently clip recall. Raise them and **all five architectures land at
> mAP@50 0.955–0.976.** The fair-resolution rematch is what surfaced the bug.

## Models

| Model | Type | Backbone | Params | Source |
|---|---|---|---|---|
| YOLOv8s | one-stage, anchor-free | CSPDarknet | 11 M | `ultralytics` |
| RT-DETR-L | transformer, NMS-free set prediction | HGNetv2 | 32 M | `ultralytics` |
| RetinaNet | one-stage, dense, focal loss | ResNet50-FPN v2 | 36 M | `torchvision` |
| FCOS | one-stage, anchor-free, FCN | ResNet50-FPN | 32 M | `torchvision` |
| Faster R-CNN | two-stage (RPN + RoI head) | ResNet50-FPN v2 | 43 M | `torchvision` |

All COCO-pretrained, head reset to 1 class, same 160/40 split, hflip aug. The
torchvision models were trained at ~800 px (default) and re-trained at 1280 px
for the rematch (`model.transform.min_size/max_size`, same everything else,
18 ep). At eval, **all per-image detection caps are raised** (`eval_all.py`).

## Results (40-image val, one torchmetrics harness, caps raised, sorted by mAP@50)

| Model | mAP@50 | mAP@75 | mAR@500 | count MAE @0.5 | best conf | **count MAE @best** | ms/img |
|---|---|---|---|---|---|---|---|
| Faster R-CNN @1280 | **0.976** | 0.955 | 0.900 | 12.8 | 0.75 | 3.0 | 106 |
| FCOS @1280 | **0.976** | **0.959** | **0.927** | 12.8 | 0.60 | 3.2 | 86 |
| FCOS @800 | 0.975 | 0.955 | 0.921 | 5.1 | 0.50 | 5.1 | 69 |
| **YOLOv8s** @1280 | 0.969 | 0.931 | 0.786 | 3.2 | 0.50 | 3.2 | **34** |
| Faster R-CNN @800 | 0.968 | 0.955 | 0.903 | 13.0 | 0.80 | **2.9** | 89 |
| RT-DETR-L @960 | 0.967 | 0.935 | 0.910 | 12.5 | 0.70 | **2.5** | 56 |
| RetinaNet @800 | 0.956 | 0.935 | 0.891 | 6.6 | 0.45 | **2.6** | 65 |
| RetinaNet @1280 | 0.955 | 0.933 | 0.888 | 4.0 | 0.55 | 3.3 | 80 |

![comparison](comparison.png)

For reference, the same torchvision checkpoints **with COCO-default caps**:
Faster R-CNN 0.839, FCOS 0.838, RetinaNet 0.881 — the numbers rounds 1–2 reported.

## Segmentation baseline — Cellpose (zero-shot)

The task's framing contrasts detection with segmentation, so: the pretrained
Cellpose `nuclei` model, **no training on BBBC039**, auto-diameter, run on the
same 40 val images. Count = number of mask labels; boxes = one per label.
Cellpose commits to a single segmentation and emits no per-object score, so
mAP@50 uses score 1.0 (its operating point) and the honest comparison is
precision / recall / F1 at IoU 0.5.

| | mAP@50¹ | P@0.5 | R@0.5 | F1@0.5 | count MAE | count bias | ms/img |
|---|---|---|---|---|---|---|---|
| Cellpose (nuclei), zero-shot | 0.838 | 0.83 | 0.92 | **0.87** | **12.6** | **+493** (+12 %) | 493 |
| *trained detectors (range)* | *0.96–0.98* | — | — | *≈0.95+* | *2.5–3.3* | *±80–260 @best conf* | *34–106* |

¹ score-1.0 operating point, not directly comparable to the detectors' swept mAP.

**Cellpose zero-shot is decisively beaten by every trained detector** — F1 0.87
vs. ~0.95+, count MAE 12.6 vs. 2.5–3.3, and it systematically **over-counts by
12 %** (recall 0.92 is fine; precision 0.83 is not — it splits nuclei the
detectors keep whole). It is also 5–15× slower (493 ms; the flow post-processing
is the cost). What it buys that detection does not: a full instance mask per
nucleus (area, shape, intensity), and it needs **zero labels** — which is the
only situation where it wins here, since BBBC039 *has* labels. A Cellpose model
*fine-tuned* on BBBC039 would be the fair segmentation-vs-detection fight and is
left for future work.

## Findings

1. **The per-image detection cap is the dominant lever — not the detector
   family, not resolution.** Raising `detections_per_img` / `topk_candidates` /
   `rpn_post_nms_top_n` from their COCO defaults lifts Faster R-CNN 0.839 → 0.968
   and FCOS 0.838 → 0.975 on the *same weights, same resolution*. That is a
   +0.13 swing from three config lines. Verified in isolation: Faster R-CNN
   averages 92 boxes/image at the default cap of 100, 118 at cap 300 — the field
   simply has more nuclei than the model is allowed to emit.

2. **With caps raised, architecture barely matters here.** Two-stage
   (Faster R-CNN), anchor-based one-stage (RetinaNet), anchor-free FCN (FCOS),
   anchor-free YOLO, and a transformer (RT-DETR) all sit in mAP@50 0.955–0.976 —
   an 0.02 spread across five fundamentally different designs. On a task of
   "find every one of many near-identical blobs," the head design is not the
   bottleneck once it is allowed to fire enough times.

3. **Resolution 800 → 1280 is a minor lever** (~+0.01 mAP@50 for Faster R-CNN
   and FCOS). It does help FCOS's *counting* (MAE 5.1 → 3.2) by tightening
   localisation. The rematch's headline: doubling resolution ≈ noise; fixing
   the detection cap ≈ +0.13.

4. **YOLOv8s now has the *lowest* recall of the five** (mAR@500 0.786 vs.
   0.89–0.93 for the others). Its value is **speed** — 34 ms, 1.6–3× faster than
   anything else — and a single-package workflow, not detection quality. On this
   dense task the torchvision models and RT-DETR recover more true nuclei.

5. **Counting accuracy is confidence-threshold-bound, per model** (unchanged
   from round 1). Optimal conf: 0.45–0.80 depending on model. At a fixed 0.5 the
   ranking is misleading (RT-DETR MAE 12.5, Faster R-CNN 13.0). After the
   per-model sweep, six of eight configs land at count MAE 2.5–3.3.

6. **RetinaNet was capped too, just less visibly** — its default
   `detections_per_img` is 300 (vs. 100 for the others) so it looked "only"
   mediocre at 0.881; uncapped it reaches 0.956. It also did **not** benefit
   from 1280 px (0.956 → 0.955) and its 18-epoch 1280 run may be undertrained.

## Significance

- **Benchmark hygiene beats model selection.** The first two rounds spent
  three training runs and a writeup concluding a detector family was unsuitable,
  when the real problem was one default value carried over from a dataset with a
  different object density. Before comparing architectures on a dense-detection
  task, audit every `*_per_img`, `*_top_n`, `topk_*`, and `max_det` in the
  inference path and set them above your worst-case object count.
- **mAP@50 still didn't predict the counting ranking.** FCOS @800 has the 3rd-best
  mAP but the worst tuned count MAE (5.1); RetinaNet @800 has the 7th-best mAP
  but ties for the best count MAE (2.6). Read the count column for a counting
  deployment.
- **The "which detector" question is close to moot for this task** once the
  caps and confidence threshold are set. Pick on **operational** grounds:
  latency (YOLOv8s), recall on crowded fields (FCOS @1280, RT-DETR), or an
  existing framework commitment. The 0.02 mAP spread is within run-to-run noise
  on a 40-image val set.
- **NMS-free (RT-DETR) is no longer a standout.** Its recall (0.910) is good but
  FCOS @1280 (0.927) and RetinaNet @800 (0.891) match or beat it once uncapped —
  the earlier "NMS discards touching nuclei" advantage was partly the other
  models being throttled upstream of NMS.
- **When labels exist, train a detector — don't reach for the off-the-shelf
  segmentation tool.** Zero-shot Cellpose (F1 0.87, count MAE 12.6) is ~4× worse
  at counting than a detector trained for 15–25 min on 160 images, and 5–15×
  slower. Cellpose earns its place only with *no* labels, or when you need the
  per-nucleus mask (area/shape/intensity) that a bounding box can't give.

## Reproduce

```bash
# torchvision @ default ~800 px
python compare/train_tv.py --model fasterrcnn --epochs 40
python compare/train_tv.py --model retinanet  --epochs 40
python compare/train_tv.py --model fcos       --epochs 40
# torchvision @ 1280 px (fair-resolution rematch)
python compare/train_tv.py --model fasterrcnn --epochs 18 --imgsz 1280 --tag 1280
python compare/train_tv.py --model retinanet  --epochs 18 --imgsz 1280 --tag 1280
python compare/train_tv.py --model fcos       --epochs 18 --imgsz 1280 --tag 1280
# RT-DETR
python compare/train_rtdetr.py --epochs 100 --imgsz 960     # converges ~ep 47
# score all checkpoints (caps raised inside eval_all.py) + plot
python compare/eval_all.py
python compare/cellpose_baseline.py    # zero-shot segmentation baseline (pip install "cellpose<4")
python compare/plot_results.py
```

Weights are gitignored. RT-DETR weights land under
`runs/detect/compare/runs/rtdetr/` (ultralytics path quirk); `eval_all.py`
globs for them. torchvision `@1280` checkpoints go to `compare/runs/<model>_1280/`.
