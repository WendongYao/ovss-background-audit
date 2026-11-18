# Contract S5: Supplementary experiment S5 — mechanism and attribution of the collision loss

> English translation of the analysis contract for stage S5, frozen on 2026-11-17 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-17. Approved by the author on 2026-11-17. The three parts are:

- Layer 1: proposition and verification of structural zeroing;
- Layer 2: per-class decomposition of Δvoc;
- Layer 3: word-by-word ablation of the LLM vocabulary.

The body of this contract was frozen before any S5 computation; it is not changed after the results are available, and deviations are appended at the end.

Motivation: the incidental observation in S4 Deviation 7 (found only after seeing the data), namely that on COCO-Object 6 methods predict 0 bed pixels under BGB and O. Layers 1 and 2 of S5 are meant to explain it; Layer 3 tests the sentence in the V4 results "a likely mechanism, but no separate ablation was done".

S5 **does not change** any data, threshold or decision of V2–S4. The PASS of V4 and the STOP of V5 remain unchanged. The vocabulary obtained in Layer 3 **must not** be reported as a new background rule or as a performance conclusion. All S5 content in the paper is written as mechanism analysis or descriptive observation.

## Layer 1: structural zeroing (frozen data only, no GPU)

**Proposition.** In the official post-processing:

- softmax is taken over all queries;
- each class takes the maximum over its own names;
- on ties, argmax returns the smallest index (as specified in the PyTorch documentation); the background class has index 0.

If the query row of class c has only one name q, and q is literally identical to some word in the background row, then the two have the same text embedding, P(background) ≥ P(c) holds everywhere, and class c is never predicted.

Caveats:

- Some methods have further processing after argmax, e.g. Trident's SAM refinement and NACLIP's PAMR. For them the proposition is only an expectation; the data decide.
- prob_thd and area_thd only change pixels to background and do not affect the proposition.

**Classification.** Based on the frozen collision lists (S2's s2_collisions.json, S3's s3_collisions.json) and the corresponding BGB class-name files. For each (method, dataset, removed background word w, corresponding query q), decide:

- **Class E (literally identical and the only name)**: after splitting by that repository's separator and stripping leading and trailing whitespace, w is literally equal to q, and the class row containing q has only this one name.
- **Class T (other)**: matches only after normalization or at the word level, or the class has other names.

**Data.**

- Original VOC21, COCO-Object: V2-P2 per-image results for BGB and O (p2_perimage_npz).
- Derived variants: S1 per-image results for BGB (s1_results).
- NC: the S2 and S3 results.

For each cell, count the total number of predicted pixels of each collision class (sum of the pred column).

**Prediction and reading.**

- P1: every E class has 0 predicted pixels in all its BGB cells.
  - All 0 → the paper writes "a literal collision makes the class impossible to predict under the official configuration" and gives the proposition;
  - Any exception → list the exceptions, rewrite the proposition as "almost impossible to predict", and state which post-processing step causes the exception.
- Also report:
  - the predicted pixels of E classes under O (if the class-name file of O is the same as for BGB);
  - the predicted pixels of E classes under NC;
  - the predicted pixels of T classes under BGB and NC. Descriptive only, no threshold.

## Layer 2: per-class decomposition of Δvoc (frozen data only, no GPU)

For each cell with NC results (the 6 COCO-Object cells of S3; each derived-variant cell of S2), compute:

- Δvoc = mIoU(NC) − mIoU(BGB) = (1/C) Σ_c ΔIoU_c, where ΔIoU_c uses dataset-level IoU.
- Assign the classes to three groups: E classes, T classes (the classes corresponding to the collisions), and the remaining classes.
- Contribution of each group = Σ_{c∈group} ΔIoU_c / C. The three groups sum to Δvoc.

Report the result for each cell and the mean over the 7 methods, and give the IoU of the E classes under BGB (expected to be 0) and under NC. Descriptive only, no threshold.

## Layer 3: word-by-word ablation of the V4 LLM vocabulary (CC) on VOC21 (GPU)

**Question.** In V4, the drop of CC relative to the 26-word rule on VOC21 is concentrated in 7 classes: ship, tvmonitor, table, person, bottle, bird, train, each with a 7-method mean drop ≥ 7.9. Can these drops be attributed to a small number of specific words in CC?

**Held fixed.**

- CC's VOC21 vocabulary: the bg_words_cc.json frozen in V4, 83 words in total.
- Rule parameters: the same as V4.
- Code: V4's p4_run.py (test mode), with only one wrapper layer added and its decision logic unchanged.
- Class set K: the 7 classes above, identified by their indices in the VOC21 unified class names.

**Stage 1 (finding words on the development set)**

- Data: VOC train (1464 images), i.e. the development set used when selecting the rule in V3; the validation set is not used.
- Methods: SCLIP, ClearCLIP, NACLIP, ResCLIP, SC-CLIP, ProxyCLIP, 6 in total. Trident would have to run SAM refinement for every variant, which is too costly, so it does not take part in Stage 1, only in Stage 2.
- One pass per method. In the same forward pass, apply V4's decision and the method's own post-processing separately for 84 variants: full CC, and 83 variants "CC with the j-th word removed". Accumulate intersect and union per class.
- Attributed words: for c ∈ K, let A_c = {w : 6-method mean development-set IoU_c(CC−w) − IoU_c(CC) ≥ 1.0}, A = ∪ A_c.
- If A is empty: the Layer 3 conclusion is "cannot be attributed to individual words", and Stage 2 is not run.

**Stage 2 (confirmation on the validation set)**

- Each of the 7 methods runs once with "CC with all words in A removed", on VOC val (1449 images).
- Recovery R_c = IoU_c(CC−A) − IoU_c(CC); drop D_c = IoU_c(SEL26) − IoU_c(CC). The last two terms are taken from the frozen V3 and V4 validation-set results. All are 7-method means.
- Decision quantity: F = Σ_{c∈K} R_c / Σ_{c∈K} D_c.

**Reading (written in advance)**

- F ≥ 0.5: write "the per-class drops of the LLM vocabulary on VOC21 are mainly attributable to a small number of specific words", list A, and manually annotate the relation of each word to its class (subclass, synonym, part, scene, unrelated).
- 0.2 ≤ F < 0.5: write "partially attributable".
- F < 0.2: write "cannot be attributed to individual words", and the statement "consistent with the collision mechanism" in the paper's text is weakened accordingly.
- Also report:
  - the changes of the classes outside K;
  - the overall mIoU of CC−A. Descriptive only; must not be written as a new rule or performance conclusion.

**Implementation check (must pass before inference)**

- On VOC val, the full-CC variant of the wrapped code must reproduce V4's SCLIP CC = 52.63 (±0.10; V4 originally ran on a 4080 SUPER).
- The per-class results accumulated by the wrapper for the full-CC variant must be exactly identical to the normal IoUMetric output of the same run.
- First run one smoke test on 20 images.
- If it fails, stop and investigate; do not relax.

**Budget and run**

- Use the RTX 3090 instance; if it has no GPU available, use a clone of it in the same region.
- GPU ≤ 6 hours, wall clock ≤ 1 day. If exceeded, first stop the slowest method in Stage 1, and record this in the deviations section.
- [internal note omitted]

## Rules

- Layers 1 and 2 only read frozen data. Layer 3 is inference only; the vocabulary, rule parameters, K, the construction of A, and the thresholds on F are frozen before computation.
- No data, threshold or decision of V2–S4 is changed.
- No password, API key or token is entered or stored.

## Deviations (appended)

1. **Layer 1 and 2 results** (2026-11-17, frozen data only; s5_l12.py bd602116, results s5_l12.json 7c95a9c1).
   - Implementation notes:
     - The query words in the collision lists are in normalized form, so when matching class rows, both sides are normalized before comparison. The first version of the script missed this step and found no T-class rows at all; after the fix it was rerun, and the second result is authoritative.
     - The class rows of original COCO-Object are taken from S3's NC class-name file. The NC file changes only the background row; the class rows are the same as for BGB.
     - ctx_bg70 has zero collisions in the S2 list and is therefore not included in the statistics.
   - **Layer 1**
     - 92 E-class rows in total, covering 17 (dataset, class) pairs and 18 cells. Background words involved: bed, curtain, fence, hill, mountain, river, road, shelf, wood.
     - **92/92 have 0 predicted pixels under BGB; P1 holds, with no exceptions.** On original COCO-Object, O is also 0 (6/6). All E classes have predictions under NC (0 rows are 0).
     - The proposition also holds for Trident (SAM refinement) and NACLIP. NACLIP uses PAMR under neither P\* nor BGB.
     - Per the contract wording: "a literal collision makes the class impossible to predict under the official configuration".
   - T classes (188 rows):
     - Under BGB, the predicted pixels are about 0.75 of those under NC (median).
     - 13 rows are 0 under BGB: wall-panel, wall-other, floor-other, ceiling-tile, etc. These classes are also rarely predicted under NC, some likewise 0; this is not structural and is descriptive only.
   - **Layer 2** (mean over the 6 methods with collisions; Δvoc = E contribution + T contribution + remaining-class contribution):

     | Dataset | Δvoc | E-class contribution | T-class contribution | Remaining-class contribution | E-class share |
     |---|---|---|---|---|---|
     | Original COCO-Object | +0.635 | +0.614 | 0 | +0.021 | 0.97 |
     | obj_bg30 | +1.631 | +1.153 | +0.186 | +0.292 | 0.71 |
     | obj_bg50 | +1.393 | +0.971 | +0.163 | +0.259 | 0.70 |

     The IoU of E classes is always 0 under BGB and 0.2–56.0 under NC; for example, bed on original COCO-Object is 45.0–56.0.

2. **Layer 3 code ready, not yet run: none of the three instances had a free GPU** (2026-11-18).
   - Scripts (s5/code, before any Layer 3 computation):
     - s5_loo.py (954dd05b): wraps the unmodified p4_run.py. The real V4 post-processing runs as usual and is used for evaluation; a copy is used to compute the 84 variants, and for every image the copy is checked against the real output.
     - s5_attr.py (bdfe50e5): constructs A according to the rule in the contract body.
     - s5_stage.sh (6334373b).
     - [internal note omitted]
   - Attempts to start an instance:
     - The RTX 3090 instance had no free GPU. [internal note omitted]
     - The other two instances could not be started. [internal note omitted]
     - No run took place.
   - Next step: wait for a GPU to become free, or have the author clone an instance with a GPU in the same region. Cloning creates a new instance and is done only after the author confirms.

3. **Layer 3 run and results** (2026-11-18, Beijing time).
   - Instance: a new instance opened by the author, cloned from the RTX 3090 instance (RTX 3090). [internal note omitted]
     - The md5 of p4_run.py (8c4aabbb) and bg_words_cc.json (b90d2007) matches the archive.
     - [internal note omitted]
   - Smoke test: 20 images, 84 variants, 0 mismatches between the copy and the real output.
   - **Implementation check passed**: SCLIP CC voc21 val = 52.64 (V4: 52.63), 0 mismatches over 1449 images.
   - **Stage 1** (02:13–03:25): all 6 methods rc=0, 84 variants, 0 mismatches throughout.
     - Development-set CC mIoU: SCLIP 52.59, ClearCLIP 56.62, NACLIP 50.22, ResCLIP 53.94, SC-CLIP 57.68, ProxyCLIP 58.92.
   - **A (by the rule in the contract body)**: 10 words in total, 73 remaining.

     | Class | Attributed words |
     |---|---|
     | ship | sailboat, tugboat |
     | tvmonitor | laptop, tv_stand, stereo |
     | table | napkin |
     | person | collar |
     | bottle | wine |
     | bird | feather |
     | train | engine |

   - **Stage 2** (03:25–04:02): the 7 methods ran CC−A with the unmodified p4_run.py, all rc=0.
   - **Computation of F**: on the server, s5_F.py (f4c43774) read the frozen V3 SEL and V4 CC validation-set results; the mIoU values read match the V3/V4 records. 7-method means:

     | Class | D_c (SEL26 − CC) | R_c (CC−A − CC) |
     |---|---|---|
     | ship | 29.97 | 23.48 |
     | tvmonitor | 13.70 | 14.83 |
     | table | 12.69 | 4.55 |
     | person | 12.21 | 2.77 |
     | bottle | 10.89 | 12.57 |
     | bird | 10.36 | 7.75 |
     | train | 7.89 | 2.29 |

     **F = 68.23 / 97.70 = 0.698 ≥ 0.5. Per the contract, write: "the per-class drops of the LLM vocabulary on VOC21 are mainly attributable to a small number of specific words."**
     - Mean change of the classes outside K: +0.38, range −0.07 (sofa) to +1.22 (aeroplane).
     - table, person and train recover only partially.
   - Overall mIoU (descriptive only; **must not** be written as a new rule or performance conclusion):
     - CC−A: SCLIP 55.55, ClearCLIP 58.99, NACLIP 53.69, ResCLIP 57.59, SC-CLIP 61.16, ProxyCLIP 63.33, Trident 66.18.
     - 7-method mean: SEL26 60.44, CC 56.00, CC−A 59.50.
   - The relation of each word to its class is annotated manually per the contract, pending the author's confirmation. Our preliminary annotation:
     - subclass: sailboat, tugboat → ship;
     - part: engine → train, feather → bird;
     - content or common accompaniment: wine → bottle, napkin → table, collar → person;
     - nearby object in the same scene: laptop, stereo, tv_stand → tvmonitor.
   - The results comprise F.log and s5_F.py. [internal note omitted]
   - [internal note omitted]

4. **Manual annotation of word–class relations: confirmed by the author** (2026-11-18).
   - Approved by the author on 2026-11-18. The preliminary annotation in Deviation 3 becomes the final annotation:
     - subclass: sailboat, tugboat → ship;
     - part: engine → train, feather → bird;
     - contents or common accompaniment: wine → bottle, napkin → table, collar → person;
     - nearby object in the same scene: laptop, stereo, tv_stand → tvmonitor.
   - None of the 10 words is a collision literally identical to a class name, so the S2/S3 collision rule (the tool) does not detect them. This is consistent with the statement in the tool's README that "synonym and part relations are outside the scope of detection".

5. **Layer 3 results archived** (2026-11-18).
   - [internal note omitted]
   - The Layer 3 results are stored in S5/results/layer3/.
