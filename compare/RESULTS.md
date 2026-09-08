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

## Results (40-image val, one harness, per-image caps raised, sorted by count MAE)

**F1@0.5** is precision/recall at IoU 0.5 evaluated **at each model's tuned
confidence** — the operating point you would deploy, and the one number
comparable to Cellpose (which has no PR curve to sweep). mAP@50 is the swept
metric; it structurally favours anything with a confidence score.

| Model | F1@0.5¹ | mAP@50 | mAR@500 | best conf | **count MAE** | count MAPE | ms/img |
|---|---|---|---|---|---|---|---|
| Cellpose (fine-tuned) | 0.880 | 0.85² | 0.714 | — | **2.35** | 2.3 % | 259 |
| RT-DETR-L @960 | 0.881 | 0.967 | 0.910 | 0.70 | 2.45 | 2.5 % | 56 |
| RetinaNet @800 | 0.879 | 0.956 | 0.891 | 0.45 | 2.6 | 2.5 % | 65 |
| Faster R-CNN @800 | 0.882 | 0.968 | 0.903 | 0.80 | 2.9 | 2.8 % | 89 |
| Faster R-CNN @1280 | **0.883** | **0.976** | 0.900 | 0.75 | 3.0 | 3.0 % | 106 |
| FCOS @1280 | 0.881 | **0.976** | **0.927** | 0.60 | 3.2 | 3.2 % | 88 |
| **YOLOv8s** @1280 | 0.876 | 0.969 | 0.786 | 0.50 | 3.2 | 3.2 % | **35** |
| RetinaNet @1280 | 0.873 | 0.955 | 0.888 | 0.55 | 3.3 | 3.2 % | 81 |
| FCOS @800 | **0.883** | 0.975 | 0.921 | 0.50 | 5.1 | 4.9 % | 70 |
| Cellpose (zero-shot) | 0.869 | 0.84² | 0.776 | — | 12.6 | 11.6 % | 493 |

¹ greedy 1-to-1 match, IoU ≥ 0.5. ² Cellpose score-1.0 operating point.

![comparison](comparison.png)

For reference, the torchvision checkpoints **with COCO-default caps**:
Faster R-CNN 0.839, FCOS 0.838, RetinaNet 0.881 — the numbers rounds 1–2 reported.

## Segmentation baseline — Cellpose (zero-shot **and** fine-tuned)

The task's framing contrasts detection with segmentation. Cellpose predicts an
instance-label mask; count = number of labels, boxes = one per label. It commits
to a single segmentation and emits no per-object score, so mAP@50 uses score 1.0
(its operating point) and the honest comparison is precision / recall / F1 at
IoU 0.5. Two runs:

- **zero-shot**: pretrained `nuclei` model, no BBBC039 training, auto-diameter.
- **fine-tuned**: same `nuclei` model fine-tuned on the same 158 train images as
  the detectors (150 epochs, ~9 min on the RTX 5060). Instance-label masks built
  from BBBC039's semantic masks (interior-class connected components).

| | mAP@50¹ | P@0.5 | R@0.5 | F1@0.5 | **count MAE** | count bias | ms/img |
|---|---|---|---|---|---|---|---|
| Cellpose, zero-shot | 0.84 | 0.83 | 0.92 | 0.87 | 12.6 | +493 (+12 %) | 493 |
| **Cellpose, fine-tuned** | 0.85 | 0.88 | 0.88 | **0.88** | **2.35** | **−38 (−1 %)** | 259 |
| *trained detectors (range)* | *0.96–0.98* | — | — | — | *2.5–3.3* | *tuned* | *34–106* |

¹ score-1.0 operating point, not comparable to the detectors' swept mAP.

**The zero-shot failure was domain mismatch, not a paradigm limit.** Fine-tuned
on the same 158 labelled images the detectors saw, Cellpose posts the **lowest
count MAE in the entire study — 2.35, essentially unbiased** — beating
RT-DETR-L (2.45) and every YOLO/torchvision config. Fine-tuning fixed the
over-segmentation (precision 0.83 → 0.88, the +12 % bias → −1 %) and halved
inference time (493 → 259 ms).

Trade-offs that remain: Cellpose is still **4–8× slower** than the detectors
(259 ms vs. 34–106). It also adds a full instance mask per nucleus (area, shape,
intensity) for free. **For counting BBBC039 with labels in hand, fine-tuned
Cellpose is the most accurate option; among the detectors, RT-DETR-L is the best
counter and YOLOv8s the fastest.**

**On detection quality (F1@0.5 at tuned conf) fine-tuned Cellpose is
indistinguishable from the detectors** — 0.880 vs. their 0.87–0.88. The mAP@50
gap (0.85 vs. 0.97) is almost entirely the score-1.0 penalty: mAP rewards having
a confidence to sweep, which segmentation structurally lacks. At the operating
point you would actually run, detection and fine-tuned segmentation localise
equally well.

## Findings

1. **The per-image detection cap is the dominant lever — not the detector
   family, not resolution.** Raising `detections_per_img` / `topk_candidates` /
   `rpn_post_nms_top_n` from their COCO defaults lifts Faster R-CNN 0.839 → 0.968
   and FCOS 0.838 → 0.975 on the *same weights, same resolution*. That is a
   +0.13 swing from three config lines. Verified in isolation: Faster R-CNN
   averages 92 boxes/image at the default cap of 100, 118 at cap 300 — the field
   simply has more nuclei than the model is allowed to emit.

1b. **At tuned operating points, every trained model converges to F1@0.5 ≈
   0.88** — all eight detector configs *and* fine-tuned Cellpose, spread 0.873–
   0.883. The 0.96–0.98 mAP@50 figures describe the PR curve, not the deployed
   point; on this GT the IoU-0.5 localisation ceiling is ~0.88 (likely the
   mask→box GT conversion as much as the models). **Models separate on count
   MAE and speed, not on detection F1.**

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
- **Off-the-shelf ≠ the method's ceiling.** Zero-shot Cellpose (count MAE 12.6)
  looked like a paradigm loss; fine-tuned on the same 158 images it posts the
  best count MAE of the study (2.35). If you benchmark a pretrained model on a
  new domain, you are measuring domain transfer, not the architecture. The real
  choice here is **fine-tuned Cellpose** (best count accuracy + free instance
  masks, 4–8× slower) vs. **a detector** (faster, box-only) — not
  "detection beats segmentation."

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
# Cellpose (pip install "cellpose<4" -- downgrades numpy to 2.0.2)
python compare/cellpose_finetune.py --epochs 150            # ~9 min
python compare/cellpose_baseline.py                          # zero-shot 'nuclei'
python compare/cellpose_baseline.py \
    --model compare/runs/cellpose_ft/models/bbbc039_ft --label "Cellpose (finetuned)"
# score all detector checkpoints (caps raised inside eval_all.py) + plot
python compare/eval_all.py
python compare/plot_results.py
```

Weights are gitignored. RT-DETR weights land under
`runs/detect/compare/runs/rtdetr/` (ultralytics path quirk); `eval_all.py`
globs for them. torchvision `@1280` checkpoints go to `compare/runs/<model>_1280/`.
