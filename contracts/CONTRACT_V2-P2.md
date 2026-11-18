# Contract V2-P2: Probe V2-P2 (stage 2) — "True progress" under a fixed evaluation setting, and background handling without tuning

> English translation of the analysis contract for stage V2-P2, frozen on 2026-11-04 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-04. Approved by the author on 2026-11-04. [internal note omitted] Upstream: Probe_OVSSProtocol01 (V2, ADVANCE_TO_DESIGN). Inference only, no training. METHOD_GO is always false. The contract was written before any new inference; it is not changed after the results are in, and deviations are appended at the end.

## The two questions to answer

- **T1 True progress:** How much of the published progress of 2024–2025 CLIP-only training-free methods relative to the earliest one, SCLIP, remains under a fixed evaluation setting?
- **T2 Background handling:** Can a background-handling rule that is not tuned for any dataset match the per-dataset-tuned background tricks in each method's official configuration?

## Methods

Verified from each official repository before running (2026-11-04):

| Method | Source | Additional models | Dataset-specific settings in the official configuration (summary) |
|---|---|---|---|
| SCLIP, ClearCLIP, NACLIP, MaskCLIP | Already implemented in V2 | None | See the V2 contract |
| ResCLIP | CVPR 2025, yvhangyang/ResCLIP | None | temp_thd and attention weights differ per dataset; no sliding window for VOC; the paper states a fixed λ |
| SC-CLIP | IEEE TIP 2025, SuleBai/SC-CLIP | None | VOC21: logit 50, threshold 0.15, area 0.1; Object: logit 55, threshold 0.35 |
| ProxyCLIP | ECCV 2024, mc-lan/ProxyCLIP | DINO ViT-B/8 | VOC21 threshold 0.2; Object 0.25; ADE short side 448 |
| Trident | ICCV 2025, YuHengsss/Trident | DINO ViT-B/16, SAM ViT-B | Short side 336/448/576 varies by dataset; threshold varies by dataset |
| CorrCLIP (optional) | ICCV 2025, zdk258/CorrCLIP | MetaCLIP, DINO ViT-B/8, pre-generated region masks | Threshold varies by dataset; backbone is not OpenAI CLIP |

Core group: SCLIP, ClearCLIP, NACLIP, ResCLIP, SC-CLIP (the CLIP-only family) and ProxyCLIP, Trident (the family that uses other vision foundation models). CorrCLIP is added only if the core group is complete and GPU usage is ≤ 48 hours.

Corroborating evidence: the ClearCLIP numbers reproduced by the SC-CLIP paper under an SCLIP-style setting are 57.0/82.3/32.5/17.3 (VOC21/VOC20/Object/ADE), consistent with the V2 framework's measured 56.96/82.21/32.47/17.35 under B_SCLIP, which indicates that the V2 framework can be reused directly.

Implementation: the new methods call the model code of each official repository directly, using minimal mmseg/mmengine stand-in classes to bypass installation dependencies; data loading, resizing, the outer sliding-window loop and the mIoU computation reuse the framework already validated in V2.

## Benchmarks

VOC21, VOC20, COCO-Object, ADE20K, exactly as in V2 (the same data and label conversion).

## Two evaluation settings

1. **Official configuration (O):** each method repository's configuration for the dataset is used as is. Used only for the reproduction gate.
2. **Fixed setting (P\*), following the unified protocol of TCL (CVPR 2023):**
   - Class names: the mmseg unified class names (identical to the class-name files of the ClearCLIP repository), the same for all methods and all datasets;
   - Prompts: 80 OpenAI ImageNet templates;
   - Short side 448, the same for all datasets;
   - Sliding window: the dataset-independent crop/stride from the method's base configuration; if there is none, 224/112;
   - No PAMR or other pixel-level post-processing (components within the method's own structure, such as Trident's SAM refinement, are kept);
   - Background: for datasets with background, only a single plain class name "background", probability threshold 0, no area threshold;
   - Method hyperparameters: only the base configuration; all dataset-specific overrides are ignored.

## Reproduction gate G0

The new methods under the official configuration are compared with the ViT-B/16 numbers in the main table of their papers (recorded before running):

| Method | VOC21 | VOC20 | COCO-Object | ADE20K | Source |
|---|---|---|---|---|---|
| ResCLIP (NACLIP base) | 61.1 | 86.0 | 35.0 | 18.0 | The "NACLIP + ResCLIP" row of Tables 1–2 of the paper; the repository's default architecture is kk attention + Gaussian neighborhood, residual removed (verified), i.e. this variant |
| SC-CLIP | 64.6 | 84.3 | 37.7 | 20.1 | Table I of the paper |
| ProxyCLIP (DINO B/8) | 61.3 | 80.3 | 37.5 | 20.2 | Table 1 of the paper |
| Trident | 67.1 | 84.5 | 41.1 | 21.9 | Table 2 of the paper |

At least 3 of the 4 benchmarks within ±1.0 counts as successful reproduction. A method that fails does not enter the "published values" comparison of T1, but its P\* results are still reported, and it is listed separately as "not reproducible under the official configuration".

## Reading

**T1 (the CLIP-only family)**

- Published progress G_rep = average over the 4 benchmarks of [the highest published value among ClearCLIP, NACLIP, ResCLIP, SC-CLIP] − [SCLIP's published value]. Published values: SCLIP 59.1/80.4/30.5/16.1, ClearCLIP 51.8/80.9/33.0/16.7, NACLIP 58.9/79.7/33.2/17.4 (no post-processing), ResCLIP (NACLIP base) and SC-CLIP as in the table above (order VOC21/VOC20/Object/ADE). Computed from these, G_rep = (5.5 + 5.6 + 7.2 + 4.0) / 4 = 5.58.
- Progress under the fixed setting G_fix: the same formula, using the measured P\* values instead.
- **T1 holds:** G_fix ≤ 0.5 × G_rep (i.e. ≤ 2.79), and the p90 of G_fix over 2,000 per-image bootstrap resamples ≤ G_rep.
- The family that uses other vision foundation models (ProxyCLIP, Trident) is computed in the same way, for description only, not part of the decision.

**T2 (background handling, VOC21 and COCO-Object only)**

- Candidate rule B\* (no tuning needed): compute the softmax with CLIP's native logit scale of 100; take, for each pixel, the maximum probability over the non-background classes; within each image, find a threshold with Otsu's method, and pixels below the threshold are classified as background. All other settings as in P\*.
- Controls: (a) P\*, plain "background", threshold 0; (b) the background tricks in each method's official configuration (class-name expansion, threshold, area threshold, logit scale) transplanted onto P\*.
- **T2 holds:** averaged over the 7 core-group methods × 2 datasets, B\* − (b) ≥ −1.0, and B\* − (a) ≥ +3.0.

**Decision**

| T1 | T2 | Conclusion |
|---|---|---|
| Holds | Holds | ADVANCE_FULL: write the full 6-month plan (audit + true-progress measurement + background method) [internal note omitted] |
| Holds | Does not hold | ADVANCE_BENCHMARK: benchmark route [internal note omitted] (audit + true-progress measurement), the method contribution is dropped |
| Does not hold | Holds | DISCUSS: the progress is mostly real, and only the background method remains; discuss with the author whether to turn this into a method paper |
| Does not hold | Does not hold | DISCUSS_STOP: only the audit of protocol differences remains, its value is weak; discuss with the author whether to stop |

Two or more of the core new methods fail G0 → CHECK_IMPLEMENTATION: investigate the implementation first, and do not tune parameters to chase the numbers.

## Budget and rules

- Wall clock ≤ 14 days; GPU ≤ 72 hours. If exceeded, cut in the order CorrCLIP → Trident, and record a deviation.
- First a smoke test on 20 images per dataset, then the formal run. Pausing at any time and resuming from checkpoints are supported (reusing the V2 mechanism).
- External weights (DINO, SAM, MetaCLIP) are downloaded from the official addresses, and their SHA-256 is recorded.
- Methods, benchmarks, the P\* definition, the B\* rule, thresholds or gates are not changed to rescue the results.

## Pre-run deviations and implementation notes (2026-11-04, appended before any inference)

1. Platform: changed to running each official repository's code directly on AutoDL (vGPU-32GB, PyTorch 2.1.2 + CUDA 12.1) (mmcv 2.1.0, mmengine 0.10.1, mmsegmentation 1.2.2). Most repositories specify mmcv 2.0.1 / mmseg 1.1.1, while Trident specifies 2.1.0 / 1.2.2; the version differences are checked by the reproduction gate. The originally planned "stand-in class" approach is no longer needed.
2. Data: VOC2012, COCO val2017 and ADE20K were taken from the AutoDL public datasets. The MD5 of COCO val2017.zip (442b8da7…) is identical to the file used in stage 1; the size of the ADE20K archive is identical to stage 1 (967,382,037 bytes); VOC is a tar.gz repackaged by AutoDL, and the validation list and annotations are checked in the results. The COCO-Stuff label maps are downloaded separately and converted to COCO-Object with the same mapping as in stage 1.
3. Method identity: in the base_config of the ClearCLIP repository, model_type is set to 'vanilla'; at run time it is set to the paper's method 'ClearCLIP' (ignore_residual=True). Trident has SAM refinement enabled (--sam_refine) following the main-result command in its README, for both O and P* (it is a component of the method itself). NACLIP's O configuration keeps its default PAMR with 10 steps (consistent with stage 1), and PAMR is turned off under P*.
4. Reproduction-gate targets: the new methods use the numbers in their papers' main tables; the O results of the old methods (SCLIP, ClearCLIP, NACLIP) serve only as a cross-check with stage 1, with the official numbers used in stage 1 as targets (README for SCLIP and NACLIP, the paper for ClearCLIP). The "published values" for T1 still follow the main text, using the papers' numbers without post-processing.
5. Per-image statistics: a subclass of mmseg IoUMetric additionally saves the intersection and union areas of each image, for the T1 bootstrap; mIoU itself is still computed natively by mmseg.
6. P\* hyperparameters of ResCLIP (found during the smoke test, appended before any formal inference): ResCLIP's base_config does not give attn_rcs_weights and attn_sfr_weights (each dataset configuration gives its own values), and the constructor raises an error directly on empty values, so "only the base configuration" cannot be carried out here. General rule fixed in advance: keys whose values are completely identical across the four official dataset configurations are treated as settings of the method itself; for required parameters whose values differ, use the paper's value where the paper states a number, otherwise the median of the four official configurations; for optional parameters whose values differ and for which the constructor has a usable default, use the default. Accordingly, ResCLIP's P\* is: delete_same_entity=True (identical in the four configurations); attn_rcs_weights=[2.0, 0.5] (τ identical in the four configurations, λ_rcs=0.5 is the optimal value given in the paper); attn_sfr_weights=[2.1, 0.7] (τ is the median of 2.2/2.1/1.8/2.1, λ_sfr=0.7 is the paper's value); temp_thd uses the default value None (varies by dataset, not given in the paper). BGB and OTSU are handled on this basis according to their original definitions. The configuration audit of the other six methods (results/cfg_audit.json) shows no problem of the same kind: the only key that is identical across the four configurations and not in the base configuration is ResCLIP's delete_same_entity.
7. Data source, supplement: the COCO-Stuff label maps were not obtained from the official address [internal note omitted]; instead, the same val2017 label maps used in stage 1 (5,000 files) were used and checked file by file against an MD5 manifest generated from the stage-1 copy, with 0 mismatches; the 1,449 VOC validation annotations and the 2,000 ADE20K validation annotations were also checked file by file against the stage-1 copies, both with 0 mismatches (results/data_check.txt).
8. Additionally installed dependencies (needed for importing each repository's code): transformers 4.37.2 (the version stated in the requirements of ProxyCLIP and Trident), fairscale 0.4.13 (imported by the BLIP code inside the ClearCLIP repository), scikit-image 0.22.0 (Trident). Weight SHA-256: ViT-B-16 5806e77c… (matches the hash in the official address), DINO ViT-B/8 575f0efc…, DINO ViT-B/16 bf34ad0f…, SAM ViT-B ec2df627…. GPU: RTX 4080 SUPER 32 GB (AutoDL vGPU-32GB).
6a. Correction to item 6 (also before formal inference): the smoke test found that ResCLIP's temp_thd is also a required parameter (the code raises an error directly when it is empty; it is not optional). Following the rule in item 6, since it varies by dataset and the paper gives no value, it takes the median of the four official configurations (0.5, 0.1, 0.11, 0.25), i.e. 0.18.
9. Rerun of out-of-memory cells (appended during the run): to shorten the wall clock, the formal run was split into two processes sharing one GPU. During this, 3 cells failed with OOM because the other process occupied about 24 GB of GPU memory: ProxyCLIP O coco_object, Trident O ade20k, NACLIP O coco_object. These are run-environment issues, unrelated to the method settings. After both processes finished, these cells were rerun separately and sequentially (run_p2.py full only fills in missing cells), with settings, data and code unchanged; the failure logs are kept in results/full_runA.txt / full_runB.txt.
