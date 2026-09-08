# Model choice was the least important decision

*A benchmark of seven detectors and segmentation models for counting nuclei in
BBBC039 — and the three setup issues that each mattered more than the
architecture.*

---

## Abstract

We trained five object detectors (YOLOv8s, RT-DETR-L, Faster R-CNN, RetinaNet,
FCOS) and a segmentation model (Cellpose, zero-shot and fine-tuned) to count
cell nuclei in the BBBC039 fluorescence-microscopy dataset, and scored all of
them through a single evaluation harness. **At each model's tuned operating
point, every trained model reaches F1@0.5 ≈ 0.89–0.90 and count MAE 1.5–2.6
nuclei/image** — a 0.014 F1 spread across two-stage, one-stage, anchor-free,
transformer, and fine-tuned-segmentation designs. The differences that *did*
move the numbers were, in order: COCO-inherited per-image detection caps
(+0.13 mAP@50 on the same weights), ground-truth box construction (F1 ceiling
0.88 → 0.90, count MAE −40%), and — a distant third — input resolution
(≈ +0.01 mAP@50). We reached the headline conclusion twice by mistake before
these were controlled for; the corrections are documented per finding below.

**Practical takeaway:** for dense small-object counting with labels in hand, the
leverage is in the evaluation harness, the label quality, and the operating-point
threshold — not the model zoo.

---

## 1. Setup

| | |
|---|---|
| **Task** | count nuclei by detection (one box per nucleus, count = #boxes) |
| **Dataset** | BBBC039 — 200 U2OS Hoechst-stained fields, 520×696 px, ~15–40 px nuclei |
| **Split** | 160 train / 40 val (seed 0). Val: 3 853 nuclei, mean 96/image, max 141 |
| **Ground truth** | watershed instance masks from BBBC039's semantic (interior/boundary) masks; boxes = tight extent, fragments < 25 px² dropped (§2.4) |
| **Detectors** | COCO-pretrained, 1-class head, hflip aug. YOLO 1280 px, RT-DETR 960 px, torchvision ResNet50-FPN ~800 px |
| **Segmentation** | Cellpose 3.1 `nuclei` — zero-shot, and fine-tuned 150 ep on the same 160 train images |
| **Hardware** | single RTX 5060 (8 GB), ~15–30 min/model |
| **Harness** | one `torchmetrics` pass: mAP@50/75, mAR@500, a per-model confidence sweep for count MAE, and **F1@0.5** (precision/recall at IoU 0.5, optimal Hungarian matching) evaluated at the swept confidence |

**Why F1@0.5 at tuned confidence is the headline metric.** mAP integrates the
precision-recall curve over all confidence thresholds; it structurally rewards
any model that emits a usable confidence score, and Cellpose does not. F1@0.5 at
the confidence you would actually deploy is the one number comparable across
detection and segmentation, and it is closer to what a counting pipeline
experiences.

---

## 2. Findings

Each finding gives the **claim**, the **path we took to it** (including wrong
turns), the **conclusion**, and a **confidence** rating given the study's
limitations (§3).

### 2.1 Architecture is not the lever

**Claim.** On this task, the detector family — two-stage vs. one-stage vs.
anchor-free vs. transformer — does not determine the outcome.

**Path.** Rounds 1–2 appeared to show the opposite: the torchvision CNNs
plateaued ~13 points below YOLO/RT-DETR (mAP@50 ~0.84 vs ~0.97), and we
concluded "two-stage and anchor-free FCN are dominated." That was wrong — see
§2.2. Once the per-image detection caps were raised and each model's confidence
tuned, the six trained models landed at:

| Model | F1@0.5 | count MAE (tuned) |
|---|---|---|
| Cellpose (fine-tuned) | 0.903 | 2.27 |
| YOLOv8s | 0.900 | 1.82 |
| Faster R-CNN | 0.899 | 1.95 |
| RT-DETR-L | 0.898 | 1.52 |
| FCOS | 0.895 | 2.38 |
| RetinaNet | 0.889 | 2.58 |

**Conclusion.** F1@0.5 spread 0.014; count MAE spread ~1 nucleus/image. For
"find every one of many near-identical blobs," the head design is not the
bottleneck once the model is configured to emit enough detections and
thresholded correctly.

**Confidence: medium.** The spread is within plausible run-to-run noise on a
40-image val set (§3). The claim is "these architectures with light tuning
converge," not a proof that architecture never matters. **Follow-up:** 3-seed
runs with confidence intervals; the official BBBC039 test split.

### 2.2 Per-image detection caps are the dominant lever

**Claim.** The single largest factor in the study was `detections_per_img` /
`topk_candidates` / `rpn_post_nms_top_n` — the number of detections a model is
permitted to emit per image.

**Path.** After rounds 1–2 blamed the ~13-point gap on architecture, we ran a
"fair-resolution rematch" (§2.3) to isolate architecture from input size. That
retrain barely moved the torchvision models — but investigating *why* surfaced
the real cause. torchvision detectors default to `detections_per_img` = 100
(Faster R-CNN, FCOS) or 300 (RetinaNet) and `topk_candidates` = 1000, values
tuned for COCO, where images hold ~7 objects. BBBC039 fields hold 90–150 nuclei.

Verified in isolation on the *same* Faster R-CNN checkpoint:

| `detections_per_img` | mAP@50 | avg boxes emitted/image |
|---|---|---|
| 100 (default) | 0.839 | 92 |
| 300 | **0.968** | 118 |
| 600 | 0.968 | 118 |

A +0.13 mAP@50 swing from one config value, no retraining. RetinaNet needed
`topk_candidates` raised too (its 300 cap plus a 1000-candidate pre-NMS limit).

**Conclusion.** Before benchmarking any detector on a dense task, audit every
`*_per_img`, `*_top_n`, `topk_*`, and `max_det` in the inference path and set it
above your worst-case object count. Two of this study's early conclusions were
artifacts of not doing so.

**Confidence: high.** Directly measured, single-variable, reproducible.

### 2.3 Input resolution is a minor lever

**Claim.** Training the torchvision models at 1280 px instead of ~800 px adds
only ≈ +0.01 mAP@50.

**Path.** We predicted (wrongly) that resolution was the ~13-point lever and
retrained Faster R-CNN, RetinaNet, and FCOS at 1280 px (`transform.min_size`).
Faster R-CNN went 0.839 → 0.848; the others moved similarly little. The
detection cap (§2.2), fixed at eval time, closed the actual gap.

**Conclusion.** Resolution matters for tiny objects in the abstract — a 20 px
nucleus at 800 px is ~12 px — but here it was swamped by the cap. It does help
*counting*: FCOS's count MAE dropped from 5.1 to 3.2 at 1280 px by tightening
localisation, before the GT rebuild (§2.4) improved all models further.

**Confidence: high** for the mAP claim; **medium** for the counting effect
(single run).

### 2.4 Ground-truth box quality set a false ceiling

**Claim.** ~1.5 of the ~12 points between the models and a perfect F1@0.5 was
label noise and a suboptimal matching algorithm, not model error.

**Path.** With the caps fixed (§2.2), all trained models converged at F1@0.5
≈ 0.88 and we attributed the ceiling to "the mask→box GT conversion as much as
the models." We then tested that:

1. **Diagnostic.** For the best model, 92% of GT nuclei had a predicted box at
   IoU ≥ 0.7 (median IoU 0.94) — localisation was excellent. The F1 gap was
   ~7% of GT nuclei with *no* matching prediction, plus greedy-matching
   collisions.
2. **GT rebuild.** Replaced the interior-connected-components split with a
   **watershed** from interior seeds over the full foreground (boundary ring
   included), a morphological close on the seeds (so a nucleus with a hairline
   interior gap doesn't split), and a real size filter (area ≥ 25 px², side
   ≥ 5 px). This dropped ~3.5% of boxes — mostly 1-px interior-class fragments.
3. **Matching.** Greedy IoU matching → optimal (Hungarian, `linear_sum_assignment`).
4. **Retrained all six models** on the new GT and re-scored.

Result:

| | old GT | new GT |
|---|---|---|
| F1@0.5 (cluster) | 0.87–0.88 | **0.89–0.90** |
| count MAE — YOLOv8s | 3.2 | **1.8** |
| count MAE — RT-DETR-L | 2.45 | **1.52** |
| count MAE — FCOS | 5.05 | **2.38** |
| YOLOv8s count bias | +84 | **−5** (over 3 853) |

**Conclusion.** GT construction moved the standings more than any architecture
choice. The F1 ceiling rose ~2 points; the residual gap to 1.0 is now mostly
genuine model error plus the still-imperfect mask→box conversion.

**Confidence: high** for "GT quality matters" and the direction of every change;
**medium** for exact magnitudes (single run per model).

### 2.5 mAP@50 does not rank the deployed task metric

**Claim.** The mAP@50 ordering and the count-MAE ordering disagree.

**Path.** Observed directly in the final table: FCOS has the highest mAP@50
(0.977) and a middling count MAE (2.38). RT-DETR-L is 5th on mAP@50 (0.960) and
**1st** on count MAE (1.52). RetinaNet is last on mAP@50 (0.950) and last on
count MAE (2.58) — so they *sometimes* agree, but not reliably.

**Conclusion.** mAP averages over confidence thresholds and IoU operating
points; a count is one threshold, one decision. For a counting deployment,
optimise and report the count metric directly. mAP is a training-progress proxy,
not the deliverable.

**Confidence: high.**

### 2.6 Confidence calibration is per-model and per-dataset

**Claim.** Each model's optimal counting confidence is different; a value copied
from a tutorial will mis-count after a model swap.

**Path.** The per-model confidence sweep found optima of **0.40** (YOLOv8s),
**0.50** (RetinaNet, FCOS), **0.70** (RT-DETR-L, Faster R-CNN). At a fixed 0.5,
RT-DETR under-counts by 161 and YOLO over-counts by 157 on the same 3 853
nuclei; at their own optima both are within ±5. Focal-loss training (RetinaNet,
FCOS) produces low, poorly-spread scores; RT-DETR's set-matching loss produces
high-confidence queries; zero-shot Cellpose has no per-object score to threshold
at all.

**Conclusion.** The confidence threshold is a per-model, per-dataset
hyperparameter that changes count MAE by 2–3× between "default" and "swept."
Sweep it on a validation set for every model.

**Confidence: high.**

### 2.7 Zero-shot performance is not the method's ceiling

**Claim.** Off-the-shelf Cellpose losing to the detectors said nothing about
segmentation as an approach.

**Path.** Zero-shot Cellpose `nuclei` scored F1@0.5 0.78, count MAE 16 (+15%
systematic over-count from over-segmentation), 452 ms/image — the worst result
in the study. We then fine-tuned the same model on the same 160 training images
(150 epochs, ~9 min). It rose to **F1@0.5 0.903 — the best of any model** — and
count MAE 2.27, essentially unbiased, at half the inference time (240 ms).

**Conclusion.** Benchmarking a pretrained model on a new domain measures domain
transfer, not the architecture. The real trade-off is: fine-tuned Cellpose
gives the best F1 and a per-nucleus instance mask (area, shape, intensity) at
4–7× a detector's latency; a detector gives a box at 33–86 ms. "Detection beats
segmentation" was never true here.

**Confidence: high** for the direction; **medium** for whether fine-tuned
Cellpose's small F1 lead is real (0.903 vs 0.900, within noise).

### 2.8 The NMS-free advantage was partly an upstream artifact

**Claim.** RT-DETR-L's recall lead in the earlier rounds shrank once the other
models were un-throttled.

**Path.** Before the cap fix (§2.2), RT-DETR-L's mAR@500 was 0.910 vs. ~0.80 for
everyone else, and we credited its NMS-free set prediction ("NMS discards a box
that overlaps a higher-scoring one, and confluent nuclei overlap"). After the
caps were raised, FCOS reached mAR@500 0.911 and RetinaNet 0.874 — the gap
narrowed to a few points.

**Conclusion.** NMS-free prediction is a real, sensible property for dense
fields, but most of the apparent advantage was the anchor-based models being
capped upstream of NMS. A cleaner test (matched detection budgets, controlled
NMS IoU) is needed to size the true effect.

**Confidence: low.** Confounded; flagged as a follow-up.

---

## 3. Limitations

- **Val set is 40 images.** The 0.014 F1 spread among trained models is within
  plausible run-to-run variance. Conclusions in §2.1 and §2.7 are suggestive,
  not established.
- **One training run per model.** No seeds, no confidence intervals.
- **Asymmetric tuning.** YOLO's augmentation and learning-rate schedule were
  tuned; the torchvision models got a light, uniform recipe. "Architecture
  doesn't matter" is really "these architectures with modest, roughly-equal
  effort converge."
- **GT is still imperfect.** The ~0.90 F1 ceiling is measured against
  watershed-derived boxes, not gold manual labels. Some of the residual is
  label error.
- **No test-set numbers.** BBBC039 has an official train/val/test split
  (100/50/50); this study used a custom 160/40 split of the full set and never
  touched a held-out test set.
- **Single dataset.** Every conclusion is BBBC039-specific (one cell type, one
  stain, one nucleus-density regime).

---

## 4. Open follow-ups

Tracked in memory (`microcc-study-todo`), promoted into §2 findings as they
complete.

| # | Follow-up | Hardens / answers | Status |
|---|---|---|---|
| A | Official BBBC039 test split + 3-seed means ± std for YOLOv8s, RT-DETR-L, one torchvision model | §2.1, §2.7 — is the convergence real or noise? | open |
| B | Cross-dataset zero-shot eval (DSB2018 / BBBC038) | generalisation — is the counter fit to this stain/density? | open |
| C | Controlled NMS-free test: matched detection budgets, swept NMS IoU | §2.8 — true size of RT-DETR's recall advantage | open |
| D | Gold labels on a 10–20 image subset (manual or SAM-assisted); re-measure the F1 ceiling | §2.4 — how much residual is model vs. label | open |
| E | Count-calibrated training: count-consistency loss or learned per-image threshold | §2.6 — remove the post-hoc sweep | open |
| F | StarDist as a second segmentation baseline | §2.7 — is fine-tuned Cellpose representative? | open |
| G | Match torchvision training effort to YOLO's (aug, schedule) | §2.1 — does the convergence survive equal tuning? | open |

---

## 5. Reproduce

Full pipeline, per-model training commands, and the raw results table are in
[`compare/RESULTS.md`](compare/RESULTS.md). Harness: `compare/eval_all.py`
(detectors) + `compare/cellpose_baseline.py` (segmentation). GT construction:
`bbbc039/masks_to_yolo.py`. Trained YOLOv8s weights: GitHub release
[v0.2.0](https://github.com/LSaiko/MicroCC/releases/tag/v0.2.0).

---

## Appendix — how the conclusions evolved

| Stage | Concluded | Corrected by |
|---|---|---|
| Rounds 1–2 | "torchvision CNNs plateau ~13 pts below YOLO/RT-DETR; input resolution is the lever." | §2.2 — COCO-default detection caps clipped recall; raising them closed the gap on the same weights. |
| Fair-resolution rematch | "Retrain torchvision at 1280 px to close the gap." | §2.3 — resolution added ≈ +0.01; the rematch is what surfaced the cap. |
| F1 convergence study | "All models converge at F1@0.5 ≈ 0.88; the ceiling is GT quality." | §2.4 — rebuilt the GT (watershed, size filter, Hungarian matching) and retrained. F1 ceiling → ~0.90, count MAE −40%. The convergence conclusion held; the numbers moved. |
