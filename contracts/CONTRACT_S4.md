# Contract S4: Supplementary experiment S4 — qualitative examples on real images

> English translation of the analysis contract for stage S4, frozen on 2026-11-17 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-17. Approved by the author on 2026-11-17. [internal note omitted]
The body of this contract was frozen before any S4 rendering; it is not changed after the results are available, and deviations are appended at the end.
Upstream: V2-P2 (per-image results for O, P\*, BGB) and S3 (per-image results for THR, NC). S4 does not change any earlier number, threshold or decision, and it produces no new quantitative conclusion.

## Purpose

Add two qualitative figures to the paper, used only to illustrate existing conclusions; the paper's text uses only the wording "illustrates".

- **Figure Q1: comparison of settings**
  - Data: 3 images each from VOC21 and COCO-Object.
  - Columns: image | GT | P\* | THR | BGB | O.
- **Figure Q2: collision**
  - Data: 3 images from COCO-Object that contain bed.
  - Columns: image | GT | BGB | NC.

## Method

Only **SCLIP** is used. This was decided before looking at any image, for the following reasons:

- it is the reference baseline for "relative progress" in Section 4.2; the other 6 methods all come after it;
- it is one of the 6 methods with the bed collision on COCO-Object.

## Image selection rule

Images are selected before any rendering, computed only from the frozen per-image statistics. No image is looked at, and no image is replaced after the images have been seen.

**Data sources**

- `archives/p2_perimage_npz.tar.gz`: O, P, BGB for SCLIP.
- `archives/s3_results.tar.gz`: THR, NC for SCLIP.

The row order of the npz files is the order of the dataset's validation list.

**Per-image mIoU**: for the classes in the image with union > 0, compute intersect/union, then take the mean. The background class 0 is included.

**Q1** (done separately for VOC21 and COCO-Object)

- Eligibility: the GT contains at least one foreground class, and background (class 0) pixels make up 20%–80% of the annotated pixels.
- Compute d_i = mIoU_i(BGB) − mIoU_i(P\*).
- Among the eligible images, take the 10th, 50th and 90th percentiles of d (numpy default linear interpolation).
- For each percentile, select the image with the smallest |d_i − q|. Ties are broken by the smallest index; an image already selected is skipped and the next nearest one is taken.

**Q2** (COCO-Object)

- Eligibility: bed (class 60) pixels in the GT make up ≥ 5% of the annotated pixels.
- Compute e_i = IoU_bed,i(NC) − IoU_bed,i(BGB).
- Percentiles and image selection as in Q1.

**Record**: the selected indices, percentiles and d/e values are written to `selection.json`, and its sha256 is recorded in the deviations section; only then does the rendering start.

**The only exception**: if an image is unsuitable for publication for content or copyright reasons, the next nearest image is taken under the same rule, and this is explained in the deviations section.

## Rendering (RTX 3090 instance)

**Inference**

- Using the same code paths and setting definitions as V2-P2 / S3, run inference on the selected images and save the predicted label maps (indexed-color PNG).
- In total 6 images × 4 settings + 3 images × 2 settings, ≤ 30 predictions.
- On the server, map the indices to file names (in the order of the dataset object) and write them to `selection_resolved.json`.

**Verification gate (every prediction must pass)**

- Recompute per-class intersect and union and compare them with the corresponding row of the frozen npz.
- A per-image mIoU difference ≤ 0.5 points passes.
- Failure: stop and investigate the row order or the setting; do not replace images and do not relax the gate.

**Budget and wrap-up**

- GPU ≤ 1 hour, wall clock ≤ 1 day.
- [internal note omitted]

## Figure making (local, no GPU needed)

- Palette: the standard VOC palette; COCO-Object uses the same generating function. Background is black, ignore is white.
- Below each cell, show the image's per-image mIoU (Q1) or bed IoU (Q2). The values come from the frozen npz.
- The caption states the selection rule and the percentile of each row.
- Output: `paper/figures/fig6_qualitative_settings.pdf`, `fig7_qualitative_collision.pdf` (and png).

## Rules

- Inference only. The selection rule, method, settings and verification gate are frozen before rendering.
- The qualitative figures are not used for any test, and the paper's text must not draw new conclusions from them.
- No password, API key or token is entered or stored.

## Deviations (appended)

1. **Selection results (by the rules in the contract body, before any rendering, without looking at any image)** (2026-11-17).
   - Scripts: s4_check.py (85157e8a), s4_render.py (6d98e473), s4_stage.sh (e6ef7c9a), select_s4.py (c9dde2f6).
   - The sha256 of selection.json is 75f6e05e960d54bc69e5034f26a8a234fac76a191ca5b58720ee8618108db473.
   - Number of eligible images: VOC21 760/1449, COCO-Object 2495/5000; COCO-Object images containing bed (≥ 5%): 137.
   - Selected indices (10th / 50th / 90th percentile):
     - Q1 VOC21: 758, 391, 63 (d = +0.24, +0.49, +0.78).
     - Q1 COCO-Object: 2832, 4844, 1430 (d = +0.06, +0.17, +0.40).
     - Q2 bed: 3974, 369, 4575 (e = +0.22, +0.56, +0.78; the bed IoU of all three images under BGB is 0).
   - Note: on VOC21 even the 10th-percentile image has a positive gain. This is the distribution of the data itself and is kept as is, per the rule.
   - Implementation detail: rendering uses s4_render.py, which wraps the original scripts that produced the frozen results (V2-P2's run_one.py for O / P / BGB, S3's s3_run_one.py for THR / NC), with only two changes: restricting the dataset indices, and additionally saving the prediction maps and GT maps. Verification is done locally with s4_check.py against the frozen npz in archives.

2. **Added supplementary figure set (proposed by the author; before any rendering)** (2026-11-17).
   - The paper's two main figures (Q1, Q2, 3 images per row) are unchanged.
   - An additional set of supplementary figures (for the appendix or supplementary material) is added, with rules identical to the contract body, except:
     - the percentiles are changed to the 5th, 15th, …, 95th, 10 images per group;
     - images already selected for the main figures are skipped.
   - New script select_s4_supp.py (e1ea61da), obtained from select_s4.py by changing only the two points above. The order was: write and commit the script first, then run it.
   - The sha256 of selection_supp.json is 451a3656d69654eac8a28264c2b7b45d4ec046acbed8c98d9b80aeec37763198. Selected indices:
     - VOC21: 14, 190, 224, 1384, 1201, 1292, 937, 254, 802, 484.
     - COCO-Object: 1651, 2626, 2812, 1486, 1082, 730, 4636, 610, 3110, 2658.
     - bed: 3209, 2284, 2168, 3172, 1142, 365, 3949, 3502, 599, 4912. The bed IoU of all 10 images under BGB is 0.
   - The same verification gate applies to the supplementary set. s4_check.py has been changed to verify both sets (1103fbc2), and s4_stage.sh also includes the supplementary set (16116fb0).
   - The inference load increases to 4 groups × 11 runs, each run ≤ 10 images; the budget is unchanged (GPU ≤ 1 hour).

3. **Change of instance and run record** (2026-11-17, Beijing time).
   - The RTX 3090 instance had no free GPU at the time. The author cloned a new instance from another existing instance [internal note omitted]; its GPU, as measured, was an RTX 4080 SUPER, not the 3090 specified in the contract.
   - Environment: mmseg 1.2.2, torch 2.1.2+cu121. The md5 of the following code matches the local archive:
     - p2/code/run_one.py (e1ced23a);
     - s1_run_one.py (d56423a0), s1_run_one_lm.py (464580cb);
     - s2_run_one.py (83851875).
   - [internal note omitted] The S3 specs, names and collisions were taken from the S3 results; the sha256 of s3_collisions.json is afe80667, identical to S3.
   - Smoke test: SCLIP P voc21, indices 758, 391, 63; per-image mIoU 2.56, 5.00, 10.66, exactly equal to the frozen values.
   - Main run: 21:26–21:32, all 20 cells rc=0, GPU time about 6 minutes.
   - The outputs comprise cells, out (prediction maps, GT maps, 39 original images) and code. [internal note omitted]
   - The gate verification (s4_check.py) was to be run locally afterwards. [internal note omitted]

4. **Gate verification results: main-figure set all passed, 2 cells in the supplementary set failed; stopped per the contract, pending the author's decision** (2026-11-17).
   - [internal note omitted] s4_check.py verified 130 (setting, image) pairs.
   - Overview of results:
     - O, P, BGB (V2-P2 path): per-image mIoU **exactly identical** to the frozen values (difference 0.000).
     - THR, NC (S3 path): very small differences, maximum |difference| 0.08, with the exception of the 2 cells below.
     - **The main-figure set (q1, q2) all passed**, maximum |difference| 0.037.
     - The per-class pixel counts of the GT labels all match, showing that the row order is correct.
   - **The 2 failing cells** (supplementary set, THR, COCO-Object):
     - Index 1486: 14.16 vs frozen 15.12 (difference −0.96).
     - Index 730: 18.25 vs 20.08 (difference −1.83).
   - Investigation:
     - The predictions of the two images differ from the frozen results in only 157 and 90 pixels respectively, about 0.03–0.07% of the image.
     - In 1486, 1 extra pixel is predicted as hair drier; in 730, 8 extra pixels are predicted as bird. Neither class is present in the GT, so the per-image mIoU gains an extra class with IoU = 0, and the score drops by 1–2 points.
     - Assessment of the cause: S3 originally ran on an RTX 3090, this run was on an RTX 4080 SUPER, and floating-point error flipped a small number of pixels near the threshold. Per-image mIoU is sensitive to single-pixel false positives.
     - Per the contract: no images are replaced, the gate is not relaxed, and the supplementary figures are not produced for now.
   - The main figures have been generated with make_qual_figs.py (281ccdc9): paper/figures/fig6_qualitative_settings, fig7_qualitative_collision (pdf/png). The values below the cells are taken from the frozen selection.json.
   - s4_check.json has been saved in S4/results/.
   - How to handle the supplementary set is for the author to decide.

5. **Handling of the supplementary set (author's decision)** (2026-11-17).
   - Decision by the author: produce the whole set of figures as usual, with the caption stating that these 2 cells differ from the frozen results in a few individual pixels. Neither the gate nor the selection is changed, and the verification conclusion remains "2 cells failed".
   - make_qual_figs.py (4334894c) reads the failing cells from s4_check.json and adds † after their values. The displayed values are still the frozen values (15.1, 20.1); the prediction maps are from this rendering.
   - Output: paper/figures/figS_qualitative_settings_voc21, figS_qualitative_settings_coco_object, figS_qualitative_collision (pdf/png).
   - The main-figure set has no failing cells, so there is no † in the main figures; the main figures were regenerated together with the script, with unchanged content.

6. **Added: rendering for option B of the overview figure, and rerunning the S3-path cells on a 3090** (2026-11-17, written before any related computation).
   - Approved by the author on 2026-11-17. The RTX 3090 instance has the same GPU model that S3 originally used.
   - **B (the full ladder on a single image, for the overview figure)**
     - Images: only the two bed images already selected for main figure Q2, indices 369 (50th percentile) and 4575 (90th percentile). Both are rendered; the author decides which one is used for Figure 0 (369 contains a child). No new images are selected.
     - Settings: P\*, THR, BGB, NC, O, with the same scripts as in the body of this contract.
     - Verification: the same gate as in the body; the per-image mIoU of each prediction differs from the frozen value by ≤ 0.5.
     - Intermediate maps: the new script s4b_render.py adds one wrapper layer, postprocess_result, on top of s4_render.py, without changing the official computation. In the same forward pass, following the official steps, it recomputes from the raw logits and saves:
       - the class probability maps P_c (after softmax, maximum over each class's names): for the background class and the classes present in the GT, float16;
       - the argmax map before thresholding;
       - the argmax map after the area threshold (only in settings that have area_thd);
       - the maximum-probability map (confidence).
     - Consistency gate: the final prediction recomputed from the saved intermediate results following the official steps must be pixel-for-pixel identical to the official output. If not, stop and make no figure.
     - The intermediate maps are only illustrative and yield no numerical conclusion.
   - **3090 rerun**
     - Using the unmodified s4_render.py, rerun all S4 THR and NC cells on the RTX 3090 instance: q1 THR × 2, q2 NC, sq1 THR × 2, sq2 NC, 6 inference runs in total.
     - Purpose: test whether the 2-cell mismatch in Deviation 4 comes from the GPU difference. The results are only recorded; the gate is not changed, and figures already made are not changed. How to annotate the supplementary figures is for the author to decide after the results are available.
   - Budget: GPU ≤ 0.5 hours. [internal note omitted]

7. **Run and verification results for Deviation 6** (2026-11-17).
   - Run: RTX 3090 instance, 23:35–23:37, all 11 inference runs rc=0.
     - [internal note omitted] The outputs are stored in S4/results/render_b/.
   - **B verification (s4b_check.py)**
     - Two images × 5 settings: maximum difference in per-image mIoU from the frozen values 0.012; all passed.
     - Intermediate-map consistency: 10/10 pixel-for-pixel identical.
     - Of these, THR and NC (S3 path; S3 also originally ran on a 3090) have per-class statistics exactly identical to the frozen values. P\*, BGB, O (V2-P2 path, originally run on a vGPU) show only tiny differences.
     - SCLIP's official configuration on COCO-Object has no area_thd, so the area-threshold step has no effect for SCLIP.
   - **3090 rerun**: all S4 THR and NC cells, 39 (setting, image) pairs in total; **for 39/39, the per-class intersect and union are exactly identical to the frozen values**, including the 2 cells that failed in Deviation 4 (1486, 730). Conclusion: the mismatch in Deviation 4 comes from the GPU difference (4080 SUPER vs 3090), not from a row-order or setting error. How to handle the supplementary figures is for the author to decide.
   - **Incidental observation** (from the frozen V2-P2 per-image results; descriptive only, not a test written in advance):
     - On COCO-Object, for the 6 methods whose background row contains "bed", under BGB and O **not a single pixel of bed is predicted** in the 5000 images; the bed IoU is always 0.
     - The same methods have bed IoU 22.9–27.6 under P\*. ClearCLIP has no collision; its bed IoU under BGB is 48.4.
     - Mechanism: the background class's score is the maximum over the background words, which include "bed", so P(background) ≥ P(bed) holds everywhere; on ties, argmax takes the background class, which has the smaller index.
     - The intermediate maps of image 4575 under BGB are consistent with this: the minimum of P(bg) − P(bed) is 0.
   - Thumbnails: make_fig0_thumbs.py generates paper/figures/fig0_thumbs/369, 4575, and two preview contact sheets (contact_*.png).

8. **Figure 0 (option B) completed** (2026-11-18).
   - Approved by the author on 2026-11-18; the choice of intermediate maps for panel (a) was left to us. Image 4575 (main figure Q2, 90th percentile) is used; it contains no child.
   - The Visio macro AddStrips adds two rows of thumbnails at the top of the original figure; the page height increases from 3.7 in to 4.45 in. The macro is kept in a separate blank drawing; the .vsdx of Figure 0 contains no macro.
     - Panel (a): input image | GT | P(background) | P(bed) | prediction. The last three are taken from the BGB setting, i.e. SCLIP's official background handling.
     - Panel (b): predictions under the five settings P\*, THR, BGB, NC, O.
   - The thumbnails are taken from render_b (3090 rendering; the verification in Deviation 7 passed). The heat-map colormap is magma, range 0–1.
   - The old version is backed up as fig0_settings_overview_v1.(vsdx|pdf|png). The new PDF has embedded fonts and no page border; the PNG was converted with pdftoppm at 600 dpi (4260×2670).

9. **Qualitative figures switched to the 3090 renderings (author's decision)** (2026-11-18).
   - Decision by the author: switch to the 3090 renderings and remove the †.
   - In make_qual_figs.py (0ce19895), the predictions in all THR and NC columns (main and supplementary figures) are replaced by the renderings from the 3090 rerun on the RTX 3090 instance (render_b/out/r3090); these renderings are exactly identical to the frozen S3 per-class statistics (Deviation 7).
   - So that results from two GPU types are not mixed within one column, the THR and NC columns of the main figures are replaced as well. The main-figure set had already passed the gate; the replacement changes only a very small number of pixels.
   - † is changed to mark only cells that still fail on the 3090; there are no such cells, so the figures no longer contain any †. The sentence about pixel differences in the supplementary-figure caption should be deleted.
   - The O, P, BGB columns still use the 4080 SUPER renderings. The per-image mIoU of these three columns is exactly identical to the frozen values (Deviation 4).
