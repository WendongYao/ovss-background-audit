# Contract V3: Probe V3 (stage 3) — Can a background-handling rule without per-dataset tuning come close to the tuned background tricks of each method?

> English translation of the analysis contract for stage V3, frozen on 2026-11-05 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-05. Approved by the author on 2026-11-05 (a low-cost check before combining directions 2 and 3). Upstream: Probe_OVSSProtocol02 (DISCUSS_STOP; in T2 the Otsu rule was on average 8.37 below the official background tricks). Inference only, no training. The contract was written before any new inference; it is not changed after the results are in, and deviations are appended at the end.

## Question to answer

Under the stage-2 fixed setting P\*, can we find **one** background-handling rule that uses the same set of values for all methods and all datasets, is **selected on the development sets only**, and on the validation sets comes within 2 points of the per-dataset-tuned background tricks in each method's official configuration (stage 2's BGB)?

## Stage-2 implementation issue found before running (recorded as is; the stage-2 results are not changed)

Trident's SAM refinement is written inside `postprocess_result`. The stage-2 OTSU cells implemented the Otsu rule by replacing `postprocess_result`, so **Trident's two OTSU cells did not go through SAM refinement**, whereas its P and BGB cells did. For the other 6 methods, `postprocess_result` only performs softmax, class merging and thresholding, and the replacement does not affect any other step. The stage-2 numbers and DISCUSS_STOP are not changed; this stage switches to an implementation that does not bypass the method's own post-processing (see "Implementation"), and reruns Trident's Otsu as a corrected reference (descriptive).

## Data splits

- **Development sets (used only to select the rule):**
  - VOC: all 1,464 images of VOC2012 `ImageSets/Segmentation/train.txt` (no overlap with the validation set);
  - COCO-Object: 2,000 images randomly drawn from COCO train2017 with a fixed seed (20261105). Labels are rasterized with pycocotools from `instances_train2017.json`: background 0, objects mapped to 1–80 by the 80 classes, drawn in order of decreasing area (small objects cover large ones), crowd regions drawn as well. This has the same source as the COCO-Stuff label maps used for the validation set (object pixels all come from the instance annotations); rasterization details may differ slightly, which affects only the development set.
- **Test sets (tested only once):** the VOC21 and COCO-Object validation sets, exactly the same as in stage 2.
- Only 3 methods are used on the development sets: SCLIP, SC-CLIP, Trident (covering an old method, a new CLIP-only method, and a method that uses other vision foundation models). All 7 methods are used for testing, 4 of which (ClearCLIP, NACLIP, ResCLIP, ProxyCLIP) were not seen during rule selection.

## Rule family (fully enumerated in advance, 120 in total)

Input: the raw cosine similarity L_q of each pixel to each query word. Query words = the unified class names (the cls file of the ClearCLIP repository, the same as stage 2's P\*) + background words.

1. **Background vocabulary V:** V0 = {"background"}; V1 = {"background"} ∪ the 26 background words in the first line of cls_voc21.txt in the SCLIP repository (sky, wall, tree, wood, grass, road, sea, river, mountain, sands, desk, bed, building, cloud, lamp, door, window, wardrobe, ceiling, shelf, curtain, stair, floor, hill, rail, fence). All datasets use the same vocabulary.
2. **Temperature T:** probability = softmax(T · L) computed over the query words in use, T ∈ {40, 70, 100}. Background probability = the maximum of the probabilities of the words in the background vocabulary.
3. **Class-presence filter E:** E0 no filtering; E(r), r ∈ {0.2, 0.4, 0.6}: the presence score of each foreground class s_c = the mean of the top 1% of that class's probabilities over the pixels of this image; classes with s_c ≥ r · max_c s_c are kept, and the remaining classes do not take part in the foreground decision.
4. **Background decision D:** D0 = argmax between the background probability and the probabilities of the kept foreground classes (no threshold); D1 = a threshold on the maximum foreground probability found with Otsu within this image (as in stage 2, 256-bin histogram), pixels below the threshold are background; D2(t) = pixels whose maximum foreground probability is below a fixed t are background, t ∈ {0.3, 0.5, 0.7}.
5. The class of a foreground pixel = the kept foreground class with the highest probability.

In total 2 × 3 × 4 × 5 = 120. No area threshold, PAMR, values that vary with image size, or any parameter taking per-dataset values is added.

## Rule selection (on the development sets only)

- Cache: the 3 development methods are run on the two development sets with the P\* setting (base configuration, short side 448, ResCLIP rule as in stage-2 deviations 6/6a — the development sets of this stage do not include ResCLIP), and for each image the raw similarity of each query word at the original image resolution is saved downsampled by a factor of 4 (bilinear, float16); labels are saved downsampled by a factor of 4 (nearest neighbor).
- Objective: the highest equally weighted average mIoU over 3 methods × 2 development sets. When differences are within 0.1, the one with "fewer components" takes priority: E0 over E(r), V0 over V1, D0 < D1 < D2, and T = 100 (used by stage-2 Otsu) takes priority.
- After selection, the rule is written to `selected_rule.json` (including the development-set scores of all 120 rules), and its SHA-256 is recorded before any test inference.

## Implementation (at test time)

The method's `postprocess_result` is not replaced. A wrapper layer is placed before it: the raw similarities are used to determine the "background pixels" and the "filtered classes" according to the selected rule, and then the similarities passed to the original `postprocess_result` are rewritten — on background pixels, the similarities of all background words are set to +10, and on foreground pixels to −10; the similarities of filtered classes are set to −10; everything else is unchanged. In this way, the method's original softmax, class merging, threshold (0 under P\*), and Trident's SAM refinement are all executed as usual, and the decisions are consistent with the rule. To keep the development-set cache consistent with the test (some methods use all query words in the forward computation), the query-word files for **both** the cache and the test use "unified class names + all V1 background words (placed in the first line as synonyms of the background class)"; V0 rules ignore those 26 words in the decision, and set them to −10 on all pixels in the rewrite.

Implementation check (before testing, must pass): with the same wrapper layer, run R0 = (V0, T=100, E0, D1, i.e. stage-2 Otsu) on the SCLIP VOC21 validation set; it must reproduce stage-2 OTSU's 49.79 (±0.05; SCLIP's similarities do not depend on the set of query words, so the additional background words have no effect). In addition, compare the SCLIP VOC development-set score computed with R0 on the development cache with the score of the same rule run on the development set at full resolution, recording only the difference caused by downsampling (descriptive). If the check does not pass, investigate the implementation and do not proceed to testing.

## Testing and reading

- Test cells: selected rule × 7 methods × {VOC21, COCO-Object} validation sets, 14 in total; in addition, Trident's R0 (through the wrapper layer, with SAM refinement) × 2 is run as a corrected reference for stage 2 (descriptive, not part of the decision).
- Δ_trick = average over the 14 cells of [rule − stage-2 BGB]; Δ_plain = average of [rule − stage-2 P\*].

| Condition | Conclusion |
|---|---|
| Δ_trick ≥ −2.0 and Δ_plain ≥ +3.0 | ADVANCE: write the full plan combining directions 2 and 3 (audit + unified protocol + tuning-free background module) [internal note omitted] |
| −4.0 ≤ Δ_trick < −2.0 | DISCUSS: the gap has narrowed but does not meet the bar; discuss with the author |
| Δ_trick < −4.0 | STOP: return to topic selection |

- Descriptive readings (do not change the decision): the difference between the selected rule's development-set score and its test score; Δ_trick for the 4 methods not involved in rule selection, separately; the split by dataset; the worst value of each cell relative to BGB; Trident's corrected R0 value.

## Budget and rules

- Wall clock ≤ 7 days; GPU ≤ 15 hours. [internal note omitted]
- Single-process run, no longer two processes in parallel (stage 2 had OOM failures because of this).
- First a smoke test on 20 images per development set to confirm that the cache and scoring code are correct, then the formal run.
- The rule family, the rule-selection objective, the test sets, the gates or the reference numbers are not changed to rescue the results; the rule is not changed after the test results are in.

## Pre-run deviations (2026-11-05, appended before any inference)

1. Hardware: [internal note omitted] the author cloned the instance to another machine in the same region, and the GPU changed from vGPU-32GB to RTX 3090 24 GB. The software environment, code, weights (SHA-256 checked one by one, all passed) and data (file counts checked, passed) were copied along with the data disk and are unchanged. The stage-2 reference numbers were obtained on vGPU-32GB; the implementation check (SCLIP R0 VOC21 = 49.79 ± 0.05) also tests the numerical differences caused by the change of GPU. GPU memory drops from 32 GB to 24 GB; if any cell fails for lack of GPU memory, it is treated and recorded as a run-environment issue, and the settings are not changed.

## Investigation of the failed implementation check (2026-11-05, appended before test inference)

2. Result of the implementation check: the wrapper-layer R0 on the SCLIP VOC21 validation set gives 49.89, which differs from stage-2 OTSU's 49.79 by +0.10, exceeding ±0.05; the script stopped as specified. Investigation (all on the RTX 3090):
   - Stage 2's original code (run_one.py) rerun on this machine: P\* is 35.91 (exactly the same as stage 2), OTSU is 49.76 (−0.03 from the change of GPU). So on the same hardware the new and old implementations differ by +0.13.
   - Diagnostic variant: changing the foreground class decision to the same fp32 argmax as in stage 2 still gives 49.89, showing that the difference does not come from class selection or from the method's own post-processing.
   - Direct comparison of SCLIP's similarities for the same input under the two query-word files: rerunning with the same file gives identical results (difference 0); between the unified class-name file and the V1 file (26 extra background words), on the shared query words the maximum difference is 3.7×10⁻⁴ and the mean difference 1.7×10⁻⁵ (mean similarity 0.22). The cause is that fp16 matrix multiplication differs slightly numerically when the number of query words differs; the contract's premise that "SCLIP's similarities do not depend on the set of query words" does not hold at the fp16 numerical level. This small difference flips the per-image Otsu thresholds and nearly tied pixels, producing an mIoU difference of about 0.1 points.
   - Conclusion: the implementation logic is equivalent to stage 2; the difference is numerical noise caused by the change in the set of query words, of magnitude about 0.1–0.15 points, far smaller than the width of the decision intervals (2 points). On this basis we proceed to testing, without changing the rule, gates or reference numbers; the results report gives ±0.15 as the numerical noise when comparing with the stage-2 numbers. The originally set ±0.05 tolerance of the check is not relaxed, and the check is recorded truthfully as "not passed, cause identified".
