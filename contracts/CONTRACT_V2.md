# Contract V2: Probe V2 — Do the gains of training-free open-vocabulary segmentation exceed the differences between the evaluation protocols themselves?

> English translation of the analysis contract for stage V2, frozen on 2026-11-02 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-02. Approved by the author on 2026-11-02. [internal note omitted] Inference only, no training. METHOD_GO is always false. The contract was written before any inference; it is not changed after the results are in, and deviations are appended at the end.

## Background (verified from the official code before running)

The official evaluation settings of the three representative methods are not the same (in each case according to the configs in the method's GitHub repository):

| Setting | SCLIP (wangf3014/SCLIP) | ClearCLIP (mc-lan/ClearCLIP) | NACLIP (sinahmr/NACLIP) |
|---|---|---|---|
| Short side | 336 | 448 | 336 |
| Sliding window (crop/stride) | 224/112 | 448/224 | 224/112; no sliding window for VOC21 and VOC20 (whole image) |
| Class-name file | Expanded names (background expanded into 26 scene words; synonyms added for person and tvmonitor) | Original class names | Same as SCLIP |
| Logit scale | VOC21 65, Object 50, others 40 | 50 | 40 |
| Background probability threshold | VOC21 0.1 (plus an area threshold of 0.1), Object 0.1 | VOC21 0.5, Object 0.4 | VOC21 0.1, Object 0.1 |
| Post-processing | None | None | PAMR enabled for the main README results (10 steps, dilations 8/16) |

The ClearCLIP paper itself also points out that SCLIP uses a "class-name rewriting trick" on VOC21; without it, the result drops from 59.1 to 51.4.

## Methods (unified implementation)

Based on the CLIP ViT-B/16 implementation in the official NACLIP code (OpenAI weights, FP32); only the last layer is changed:

| Method | Last layer |
|---|---|
| MaskCLIP | Residual removed, FFN removed, attention set to the identity matrix (value only) |
| SCLIP | Residual and FFN kept; attention = softmax(qqᵀ) + softmax(kkᵀ) |
| ClearCLIP | Residual removed, FFN removed; attention = softmax(qqᵀ) |
| NACLIP | Residual removed, FFN removed; attention = softmax(kkᵀ + Gaussian neighborhood term, σ = 5) |

## Protocol bundles (every method is run once under every bundle)

B_SCLIP, B_ClearCLIP and B_NACLIP each take the official settings from the table above. Shared by all: 80 OpenAI ImageNet prompt templates; images resized bilinearly with mmseg Resize(scale=(2048, short side), keep_ratio); logits upsampled bilinearly back to the original image size before taking the argmax; mIoU following mmseg IoUMetric (intersection and union accumulated over the whole dataset, 255 ignored, classes with a union of 0 not counted).

## Data (4 benchmarks)

- VOC21, VOC20: PASCAL VOC 2012 val, 1,449 images; VOC20 sets background to ignore.
- COCO-Object: COCO 2017 val, 5,000 images; generated from the COCO-Stuff stuffthingmaps using the mapping in cvt_coco_object.py from the SCLIP repository (80 object classes + background; stuff and unlabeled are both assigned to background).
- ADE20K: ADEChallengeData2016 val, 2,000 images, 150 classes.
- Context and Cityscapes are not included (the conversion tools or the download require a login); this is a scope limitation, not a selection.

## Reproduction gate G0

Each method, under its own official bundle, is compared with the officially reported numbers (SCLIP and NACLIP: their respective READMEs; ClearCLIP: Tables 2/3 of the paper):

| Method | VOC21 | VOC20 | Object | ADE20K |
|---|---|---|---|---|
| SCLIP (README) | 59.63 | 81.54 | 33.52 | 16.45 |
| ClearCLIP (paper) | 51.8 | 80.9 | 33.0 | 16.7 |
| NACLIP (README, with PAMR) | 64.07 | 83.03 | 36.19 | 19.05 |

A method counts as successfully reproduced only if at least 3 of the 4 benchmarks are within ±1.0 mIoU. A method that fails reproduction does not enter V2-a or V2-b. If 2 or more of the three methods fail → CHECK_IMPLEMENTATION: stop, and do not tune parameters to chase the numbers. MaskCLIP has no official bundle of its own and serves only as a reference arm.

## Reading (using only those of SCLIP, ClearCLIP and NACLIP that pass G0)

- Protocol variation s(method, benchmark) = maximum − minimum of the method's mIoU across the three bundles.
- Method gap g(benchmark, bundle) = median of the pairwise |mIoU differences| between methods within the same bundle.
- Published ranking = the ordering, on each benchmark, of the published numbers of each method under its own official bundle (table above).

| ID | Condition | Meaning |
|---|---|---|
| V2-a | Median of s over all (method, benchmark) ≥ median of g over all (benchmark, bundle) | The change brought by the protocol is no smaller than the gap between methods |
| V2-b | Among the 12 (benchmark, bundle) cells, those in which the method ordering differs from the published ranking ≥ 50% | The published ranking often does not hold under a unified protocol |

Decision: V2-a and V2-b both hold → ADVANCE_TO_DESIGN (expand to about 10 methods, add Context, Cityscapes and ViT-L, and propose a unified protocol and evaluation framework). Otherwise DROP. G0 fails → CHECK_IMPLEMENTATION.

Descriptive outputs (not part of the decision): each method's "advantage of its own bundle relative to the other methods' bundles"; single-factor decomposition (the individual effects of class names, threshold, resolution/sliding window, and PAMR).

## Budget and rules

Data are downloaded and converted in a Linux environment on the author's computer; inference runs on the author's RTX 3090 Ti, wall clock ≤ 4 hours, first a smoke test (20 images per benchmark) and then one formal run. Methods, benchmarks, bundles, thresholds or gates are not changed to rescue the results. Code sources: the CLIP and PAMR implementations of NACLIP (MIT) and SCLIP (MIT), with their copyright notices retained.

## Pre-run deviations (2026-11-02, appended before any GPU inference)

1. Precision: checking the official code showed that all three repositories run in FP16 on the GPU (SCLIP and NACLIP use OpenAI clip.load, which does not convert to FP32 on CUDA; ClearCLIP explicitly sets precision='fp16'). For faithful reproduction, inference is changed to FP16, and text features are also encoded in FP16; PAMR and mIoU accumulation use FP32. This overrides the one mention of "FP32" in the main text.
2. For ADE20K, the class-name files of the two repositories are completely identical (same MD5), so the C set and the S set are computed only once.
3. Text tokenization is done in the cloud (the same OpenAI BPE tokenization and 80 templates as the official code), and only the token ids are passed to the GPU machine; the template strings are parsed as text, and no repository code is executed.

## In-run deviation: budget (appended 2026-11-02 14:40; no formal results had been produced at this point)

The actual speed of the formal run was far below the estimate: all 12 VOC inference passes took 6,242 seconds, and a single COCO-Object pass took about 4,160 seconds. Extrapolating from this, completing everything would take about 14–16 hours, exceeding the 4-hour wall-clock budget written in the contract. Handling: the run is not interrupted, the scope is not changed and the samples are not reduced; the run is allowed to complete with the original settings. The overrun concerns computation time only and does not involve any decision made after seeing results. The main causes of the slowness are that every image is scored under all of 2 class-name sets × PAMR on/off × several post-processing variants (for the descriptive single-factor decomposition), and that PAMR runs at the original image resolution.

## In-run deviation: crash of the first formal run (appended 2026-11-03 04:55; no formal results were produced or viewed before the crash)

- The first formal run, after completing all of VOC and COCO-Object (36,792 seconds), hit a large image in the first ADE20K pass and crashed with a GPU out-of-memory error in PAMR. Results were written only at the very end, so no formal results were saved to disk. The log is retained as v2_full_log_attempt1_oom.txt.
- Fix 1 (consistent with the official code): when PAMR raises a RuntimeError (out of GPU memory), the official NACLIP predict() keeps the logits without PAMR and continues; our framework omitted this, and it is now added following the official code, with each image that falls back recorded.
- Fix 2 (affects only the amount of computation): only the cells needed by the frozen reading are computed, i.e. the main cells of the three official bundles plus the single-factor substitution cells starting from each method's own bundle, 103 in total; the remaining combinations are no longer computed. The definitions of V2-a, V2-b and the descriptive outputs are unchanged.
- Fix 3 (engineering): a checkpoint is written each time a (data group, method, pass) is completed, so that after a crash the run can resume from the checkpoint.
- After these changes, one complete rerun is performed, including the smoke test; methods, benchmarks, protocol bundles, thresholds and gates were all unchanged.
