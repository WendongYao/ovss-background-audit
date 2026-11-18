# Contract S1: Supplementary experiment S1: Relation between the protocol gap and the background pixel fraction

> English translation of the analysis contract for stage S1, frozen on 2026-11-07 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-07. Status: **Frozen (2026-11-07)**. Approved by the author on 2026-11-07. The body was frozen before any S1 computation; the "Pre-freeze clarifications" were written at the time of freezing; nothing is changed after results come out, and deviations are appended at the end.
Upstream: V2-P2 (O, P\*, BGB on VOC21/VOC20/COCO-Object/ADE), V3 (SEL26), V5 (Context-60, Q1 = STOP). S1 does not rerun earlier tests, does not change earlier thresholds, and does not overturn the STOP of V5. It answers only one new question, for use in the journal paper.

## Questions

- **H1 (main question):** Across datasets and controlled derived data, does the gap produced by the background trick increase monotonically with the background pixel fraction?
  - The gap is measured as Δ_BG = BGB − P\*, averaged over the 7 methods.
- **H2 (per image, zero GPU):** Within the same dataset, do images with a higher background fraction also have a larger O − P\*?

Why the main metric is BGB − P\* rather than O − P\*: derived datasets have no "official configuration"; BGB transplants only the background trick, which isolates exactly the background factor. O − P\* is reported only on datasets that have an official configuration, as an auxiliary metric.

## Definition of the background pixel fraction

- For a validation split: bg_frac = (number of pixels labeled as background) / (number of all non-ignored pixels).
  - Background: label 0.
  - Ignored: label 255.
  - Computed on the annotation maps at original resolution, accumulated over the whole split.
- The per-image bg_frac_i has the same definition, computed only within a single image.
- Datasets without a background class (VOC20, ADE20K with 150 classes, Context-59) have bg_frac recorded as 0; they serve only as reference points and do not enter the H1 correlation computation (BGB does not exist on them).
- The script is written first and its md5 recorded, then it is run. Output: bgfrac_report.json.
- Check: Context-60 must give 12.6% (±0.1, V5's ctxdata_report.json). If it does not match, stop and investigate.

## Data points

**Existing points (no GPU needed):** BGB − P\* for VOC21, COCO-Object and Context-60, from the V2-P2 results table and the V5 results. Their bg_frac is computed in Tier 1 (for VOC21 and COCO-Object it is currently unknown).

**Controlled derived points (GPU needed):** Constructed in the same way that COCO-Object is derived from COCO-Stuff: a subset of foreground classes is merged into background, these classes are deleted from the query class list, and all other settings are unchanged.
- **Ctx-bgK:** Starting from Context-60, classes are merged into background one at a time in descending order of pixel frequency on the Context-60 validation set (ties broken by class index), until bg_frac first becomes ≥ the target value. The target values are 30%, 50% and 70%, giving 3 variants in total.
- **Obj-bgK:** Starting from COCO-Stuff-171. COCO-Object equals "all stuff merged into background". In addition, intermediate variants are taken in which a subset of stuff classes is merged into background: in descending order of stuff-class pixel frequency, stuff classes are merged into background one at a time until bg_frac first becomes ≥ each of the following two targets:
  - 30%;
  - (30% + bg_frac of COCO-Object) / 2.
  - 2 variants in total. If the bg_frac of COCO-Object is itself < 40%, only the 30% variant is taken. Stuff classes that are not merged remain as query classes.
- The merge lists are determined solely by the frequency rules above; they are output in Tier 1 and frozen, and only then does the GPU stage begin. No method's predictions are looked at.
- Consistency check at K = 0: Context-60 itself under the Ctx-bg rule must reproduce V5's SCLIP P\* = 30.41 and BGB = 31.93 (±0.05, on the same RTX 3090 instance).

**Definition of BGB on derived data (written in advance):**
- Ctx variants use each method's official background trick for Context-60; Obj variants use the official background trick for COCO-Object.
- The transplant is done exactly as in V2-P2 / V5 (the code run_one.py is unchanged).
- Known confound: merging classes also reduces the number of query classes; this is stated as a limitation in the paper.

## Tiers and cells

| Tier | Content | Cells | GPU |
|---|---|---|---|
| Tier 1 | bg_frac of each dataset and derived scheme (CPU); H2 per-image analysis, using existing per-image npz | 0 | 0 (CPU-only mode, no GPU attached) |
| Tier 2 | Ctx-bg30/50/70 × 7 methods × {P\*, BGB}, plus 1 cell for the K = 0 consistency check | 43 | about 13 hours |
| Tier 3 | Obj intermediate variants (1–2) × 7 × {P\*, BGB} | 14–28 | about 5–9 hours |
| Tier 4 (optional) | SEL26 × 7 on the Tier 2 + 3 variants | 35 | about 11 hours |

- Basis of the GPU estimate: V5 used about 14.5 hours for 49 cells on the RTX 3090 instance used for V5 (excluding idle time), i.e., about 0.3 hours per cell on average. The derived data have the same images and are of the same order of magnitude as Context-60 / COCO, so we estimate 0.3 hours per cell; after the smoke test, the estimate is updated from measurements and recorded as a deviation.
- **Hard limit:** Tiers 1–3 combined GPU ≤ 25 hours; wall clock ≤ 5 days. If exceeded, cut in the order Tier 4 → Tier 3 → Ctx-bg70.
- [internal note omitted]

## Decision (written in advance)

**H1:** The point set is VOC21, COCO-Object, Context-60 + the Tier 2/3 variants that are completed (at most 8 points).
- **SUPPORTED** requires all three of the following:
  - (a) Spearman ρ between bg_frac and mean Δ_BG ≥ 0.8;
  - (b) the slope of a linear fit is > 0, and the 90% interval of 2,000 bootstrap replicates over methods (each replicate resamples the 7 methods and then takes the mean) does not contain 0;
  - (c) within the Ctx-bg series (Context-60, bg30, bg50, bg70), for ≥ 5 of the 7 methods, Δ_BG is monotonically non-decreasing in bg_frac.
- **PARTIAL:** (a) or (b) holds, but not all three.
- **NOT_SUPPORTED:** neither (a) nor (b) holds.
- With fewer than 5 points (e.g., if Tier 2 fails), the conclusion is INSUFFICIENT, and only a description is given.

**H2:** On VOC21, COCO-Object and Context-60, the images of each dataset are separately split into 5 equal-frequency bins by bg_frac_i; in each bin, the mIoU of O and of P\* is computed from accumulated intersections and unions.
- Condition for H2 to hold: the Spearman ρ between bin index and the 7-method mean (O − P\*) is ≥ 0.9, and this is satisfied on at least 2 of the three datasets.
- Only the existing per-image npz are used; no new inference is run.

**How it is used in the paper:**
- SUPPORTED: the paper may state "the protocol gap grows with the background fraction", with Fig. 2 as the main figure.
- PARTIAL: only a descriptive relation is written; the title does not use "track the background".
- NOT_SUPPORTED / INSUFFICIENT: this claim is removed, and only the gap on each dataset is reported.
- No outcome rewrites the STOP of V5.

## Data sources and verification

- **The RTX 3090 instance used for V5**: Context-60 annotations converted from val.pth (val.pth md5 92636c23dca1b3f0ef8b1fba3a9e6b9e), VOC2012, COCO val2017 and stuffthingmaps, code and weights of the 7 methods. After startup, first verify:
  - the val.pth md5;
  - the count of 5104 annotation PNGs;
  - the 5,000 COCO stuffthingmaps files, compared file by file against the md5 list in V2-P2's data_check.txt.
  - If anything does not match, stop; do not re-download and do not replace.
- **Per-image npz (H2):**
  - Context-60: in the V5 result package [internal note omitted].
  - VOC21 / COCO-Object: in the V2-P2 per-image npz package [internal note omitted].
  - If this package is not available on the RTX 3090 instance, H2 is done on Context-60 only, or on a copy of the package verified by md5 [internal note omitted].
- **No external dataset is newly downloaded.** Cityscapes requires a login, and its 19-class evaluation has no background class, so it is not included.

## Implementation checks (must pass before the GPU stage)

- K = 0 consistency check: SCLIP Context-60 P\* = 30.41, BGB = 31.93, ±0.05.
- Derived annotation checks:
  - the number of classes in each variant equals 60 − the number of merged classes;
  - bg_frac agrees with the Tier 1 report (±0.01%);
  - the total number of pixels is the same as in the original data.
- If a check fails, investigate the implementation; do not relax the tolerances.

## Time constraints and failure handling

- [internal note omitted]
- **Suggested schedule (after approval by the author):**
  - Day 1: start the RTX 3090 instance in CPU-only mode, run Tier 1, and at the same time verify the data [internal note omitted];
  - Days 2–3: run Tier 2 in GPU mode;
  - Day 4: Tier 3;
  - the GPU stage ends on November 20 at the latest [internal note omitted].
- [internal note omitted]
- **GPU shortage:** the cloud region used often has no free GPUs.
  - First wait, for at most 24 hours.
  - Then use "cross-instance data copy" to migrate to an instance in the same region that has a GPU; after the migration, redo the implementation checks, and record this as a deviation.
  - If no GPU can be obtained by November 18, stop the GPU stage and use only the Tier 1 results; H1 is judged INSUFFICIENT or described from the existing points.
- **OOM or script error:** record the log; after fixing, rerun only the missing cells and keep the failure logs (the same procedure as V2-P2 deviation 9).
- [internal note omitted]

## Rules

- Inference only. The merge rules, target fractions, decision thresholds and the BGB transplant method are frozen before the GPU stage.
- The data, thresholds, seeds and criteria of earlier tests are not changed; V2–V5 are not rerun to rescue any result.
- No password, API key or token is entered or stored.

## Pre-freeze clarifications (2026-11-07, written before any S1 computation)

1. **Scope**: Tiers 1–3 only. Tier 4 (SEL26) was not approved and is not run.
2. **Merge rule for Ctx-bgK**: as in the original body text, all 59 foreground classes are ranked by pixel frequency, and the merge order does not distinguish between stuff and thing classes. The merge list is output by Tier 1, and the GPU stage begins only after it is frozen.
3. **Implementation of the derived datasets**:
   - Annotations are renumbered to compact indices: 0 is background, and the remaining classes keep their original relative order.
   - The datasets are registered as subclasses of the parent dataset classes (PascalContext60Dataset / COCOObjectDataset), replacing only the class names in METAINFO. Both the images and the image lists are symbolic links to the original data.
   - In the COCO-derived annotations, the original 255 (unlabeled) is assigned to background, consistent with the COCO-Object conversion.
4. **Query class lists of the derived datasets**:
   - **P\***: Ctx variants use the unified class-name file cls_context60.txt with the rows of the merged classes deleted. Obj variants use cls_coco_object.txt, followed by the rows of the unmerged stuff classes from cls_coco_stuff.txt, appended in their original order.
   - **BGB**: Ctx variants use each method's official class-name file from cfg_context60 with the rows of the merged classes deleted. Obj variants use each method's official cls_coco_object file, followed by the same unmerged stuff rows, likewise taken from the unified cls_coco_stuff.txt.
   - The other background keys (prob_thd, area_thd, logit_scale) take the values of the parent dataset's official configuration, the same as in the V2-P2 / V5 transplant.
   - Known confound (to be written into the paper's limitations): some methods' official background rows contain words such as sky and grass, which may coincide with the names of unmerged stuff queries.
5. [internal note omitted]
6. **Rescue of results (not part of the S1 decision)**:
   - The earlier instance holding the V2-P2 / V4 raw results and the RTX 3090 instance were both started in CPU-only mode and checked.
   - The digests of the md5 lists of p2/results (1105 files) and p4 (120 files) on the earlier instance are exactly identical to those on the RTX 3090 instance: 7ee5d01d…, 988c7512….
   - [internal note omitted]

## Deviations (appended)

1. **Floating-point error in the H2 decision** (2026-11-07, recorded after the Tier 1 results came out; the decision threshold is not changed).
   - For VOC21, the Spearman ρ between bin order and mean gap, computed exactly from the ranks, is 0.9: the ranks are 1,2,3,5,4, ρ = 1 − 6·2/(5·24) = 0.9, which meets the threshold ≥ 0.9.
   - However, the floating-point value returned by scipy is slightly smaller than 0.9, so the script recorded VOC21 as not passing (n_datasets_pass = 0).
   - By the exact value, the passing datasets are 1/3 (VOC21); for COCO-Object and Context-60, ρ is 0.7 in both cases.
   - The H2 conclusion is unchanged: **NOT_SUPPORTED** (requires ≥ 2/3). tier1_h2.json is kept as is and is not rewritten.
2. **Tier 1 complete; GPU stage not yet started** (2026-11-07, about 16:40 Beijing time).
   - Tier 1 was completed on the RTX 3090 instance in CPU-only mode.
   - The derived data and query class lists were generated, and the checks passed:
     - the bg_frac of each variant differs from the Tier 1 report by 0;
     - the total number of pixels is unchanged;
     - the class counts are 57 / 54 / 49 / 60 / 165 / 151.
   - The frozen merge-list file is tier1_bgfrac_frozen.json, sha256 cc5d61b6….
   - After the CPU-only session was shut down, the instance could not be started with a GPU because no GPU was free on its host [internal note omitted].
   - Following the rule in the body, we first wait, for at most 24 hours, retrying at intervals during that time; only after that is cross-instance migration considered.
   - [internal note omitted]
3. **Switch to a cloned instance (cloned by the author)** (2026-11-07, about 20:30 Beijing time, before any S1 GPU inference).
   - Reasons:
     - The author instructed us to use the cloned instance.
     - [internal note omitted]
   - The cloned instance was cloned from the earlier instance holding the V2-P2 / V4 raw results; its GPU is an RTX 4080 SUPER 32 GB (vGPU), not the RTX 3090 of the original instance.
     - At this time the RTX 3090 instance had GPUs available, but as instructed by the author, the cloned instance was still used.
   - The cloned instance had neither the converted Context-60 data nor the S1 data, so the following was redone on it:
     - Context-60 was converted with the same val.pth and the same p5_ctxdata.py. val.pth md5 92636c23…, p5_ctxdata.py md5 2a9883f1…, both identical to V5; the script likewise exited at the assertion "5105 required", having written 5104 images, the same as V5 deviation 4.
     - The md5s of the 4 S1 script files are the same as on the RTX 3090 instance: s1_tier1 ffeb144f, mk_s1_specs 41a9c0b0, s1_run_one d56423a0, s1_stage 9e772ff2.
     - Tier 1 stage a was rerun. We continue only if the sha256 of its output equals the frozen value cc5d61b6…; otherwise we stop.
   - The thresholds of the implementation check are not relaxed: SCLIP ctx_bg0 P\* 30.41, BGB 31.93, ±0.05. These two reference values come from the RTX 3090 used in V5; if fp16 numerical differences on the 4080S cause the check to exceed the tolerance, we stop according to the contract and do not relax it.
   - [internal note omitted]
4. **Implementation check passed; GPU stage started** (2026-11-07 20:51, Beijing time).
   - On the cloned instance (RTX 4080 SUPER), SCLIP ctx_bg0: P\* 30.40 (reference 30.41), BGB 31.92 (reference 31.93), both within ±0.05.
   - The smoke test passed: SCLIP P ctx_bg70, 20 images, 12.38; Trident BGB obj_bg50, 20 images, 18.48. Smoke-test results do not enter the decision.
   - Measured time per cell is about 7–8 minutes (SCLIP), faster than the budgeted 0.3 hours.
5. **H1 scoring implementation details** (written before the results came out).
   - Bootstrap random seed 0 (numpy default_rng(0)). The body did not specify a seed; it is fixed here.
   - The "linear fit" uses least squares: y = mean Δ_BG, x = bg_frac.
   - BGB and P\* values for the three existing points:
     - VOC21 and COCO-Object are taken from V2-P2 (the V2-P2 results table);
     - Context-60 is taken from V5 (the Q1 table of the V5 results).
   - The bg_frac of the existing points is taken from the values computed in Tier 1.
   - The scoring script is experiments/S1/code/s1_score.py.
6. **10 Tier 3 BGB cells ran out of GPU memory (OOM) and were rerun with a low-memory equivalent implementation** (2026-11-08, about 13:50 Beijing time, recorded before looking at any S1 scoring result).
   - **Failed cells**: on the two variants obj_bg30 and obj_bg50, the BGB cells of the five methods SCLIP, NACLIP, ResCLIP, SC-CLIP and ProxyCLIP, 10 in total, all exited with rc=1.
   - **Cause**: the background rows in these methods' official COCO-Object class-name files carry many synonyms; together with the 84 and 70 stuff queries kept after merging, `(X * cls_index).max(1)[0]` in each method's own postprocess_result creates a [number of classes, number of queries, H, W] tensor in a single step, which needs about 38.6 GiB and exceeds the 32 GB of GPU memory. The failure logs were renamed to *_oom.log and kept.
   - **Fix (affects memory only, does not change results)**: a new wrapper s1_run_one_lm.py (md5 464580cb).
     - Before running, it rewrites the line above into a per-class loop: for each class, the same element-wise multiplication is done, followed by the same max.
     - s1_run_one.py itself is not changed.
     - Checked on the 4080S with random tensors (float32 and float16, up to 165 classes × 200 queries): bitwise equal (torch.equal is True).
     - Exactly 1 line was replaced for each of the 5 methods.
   - **Rerun scope**: only these 10 missing cells (rerun_lm.sh); the other 60 cells are not rerun.
   - **GPU usage**: the main run used about 15.9 hours up to 12:28; the rerun is expected to take about 5–6 hours; the total is still within 25 hours.
7. **Rerun slower than expected; a budget-deadline watchdog was added on the machine** (2026-11-08 16:41, Beijing time; still no scoring result has been looked at).
   - The per-class loop makes the rerun cells markedly slower: SCLIP 1245 seconds; ResCLIP 6643 seconds. As of 16:41, 4/10 rerun cells were complete.
   - A watchdog script deadline.sh was added on the machine: at 11-08 21:36 (the 25-hour GPU limit), if the rerun has not finished, the remaining rerun is stopped and this is written to rerun_log.txt.
   - Cells stopped in this way are treated, according to the contract, as cut Tier 3 cells: the corresponding variant lacks BGB and does not enter the H1 point set.
   - Thresholds and decision rules are unchanged.
8. **Run completed** (2026-11-08 20:43, Beijing time).
   - All 10 rerun cells finished with rc=0; deadline.sh was not triggered, and no cell was cut. The 70 cells plus 2 check cells are complete.
   - Time used:
     - the compute time of the successful cells totals about 22.8 hours (excluding failed cells, smoke tests and idle time);
     - the wall clock from 11-07 20:36 to 11-08 20:43 is about 24.1 hours.
     - Both are within the 25-hour limit.
   - [internal note omitted]
   - The md5 of cells.csv is 8a2492b9… [internal note omitted].
   - [internal note omitted] For the scoring, see the S1 results.
