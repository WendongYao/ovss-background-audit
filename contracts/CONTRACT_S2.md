# Contract S2: Supplementary experiment S2: splitting the official background trick into a "threshold" part and a "background vocabulary" part, testing the "collision" explanation

> English translation of the analysis contract for stage S2, frozen on 2026-11-08 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-08. Status: **frozen (2026-11-08)**. Approved by the author on 2026-11-08. The body was frozen before any S2 computation; it is not changed after results are available, and deviations are appended at the end.

Upstream: S1 (H1 SUPPORTED; BGB − P\* is negative on the COCO-derived variants: obj_bg30 mean −2.59, obj_bg50 mean −0.18; see reading 3 of the S1 results). S2 does not rerun, and does not change, any S1 cell, threshold or decision.

## Question

On the COCO-derived variants that retain stuff queries, the transplanted official background trick (BGB) lowers mIoU. Which of the following is the cause?
- the background synonyms "collide" with the retained query class names (the S1 hypothesis),
- or numerical settings such as the threshold and the logit scale are unsuitable?

## Settings (two new settings on top of S1's P\* and BGB)

The trick is split into two parts: the numerical part is the official prob_thd, logit_scale, area_thd (whichever of these exist; likewise below); the vocabulary part is the background line of the official class-name file.

- **THR (transplant only the numerical part)**: P\* plus the numerical part of the parent dataset's official configuration. The class-name file is S1's P\* file, i.e., the background is a single plain "background".
- **NC (transplant the full trick with colliding words removed)**:
  - Identical to S1's BGB, except that the "colliding words" are deleted from the official background line. The collision determination is given below.
  - If the background line is empty after deletion, "background" is used.
  - All lines other than the background line are the same as in BGB, and the numerical part is also the same as in BGB.
- The controls are taken directly from existing S1 cells, without rerunning: S1's P\* and BGB.

## Collision determination (written in advance; depends only on class-name strings and does not look at any prediction)

1. **Splitting**: the background line and each retained query line are split with the method's own separator, consistent with SEP in V4's p4_run.py: ";" for ProxyCLIP and Trident, "," for the others. After splitting, leading and trailing whitespace is stripped.
2. **Normalization**:
   - convert to lowercase;
   - replace "-" and "_" with a space;
   - if the last word is "other" or "merged", remove it;
   - strip leading and trailing whitespace.
3. **Determination**: a background word w collides with a retained query word q if and only if, after normalization, any of the following holds:
   - w == q;
   - w is one of the space-separated tokens of q;
   - q is one of the space-separated tokens of w.

   For example, w = sky collides with q = sky-other, and w = wall collides with q = wall-concrete.
   - The word "background" itself is never deleted.
4. **Only colliding words in the background line are deleted**; the query lines are not changed in any way.
5. The script mk_s2_names.py generates, for each method and each variant, the NC class-name file and the collision list (s2_collisions.json, with its sha256 recorded), fixed before any S2 inference.

## Cells

| Tier | Variant | Settings | Cells | Estimated GPU |
|---|---|---|---|---|
| A | obj_bg30 | THR, NC × 7 methods | 14 | about 6 hours |
| B | obj_bg50 | THR, NC × 7 methods | 14 | about 6 hours |
| C | ctx_bg70 (control on which the background trick is positive) | THR, NC × 7 methods | 14 | about 3 hours |

- When a method has 0 colliding words on a variant, the inputs of NC and BGB are identical: this cell is not run; the S1 BGB value is taken directly and marked "zero collisions".
- Basis of the estimate: the measured run times of S1 on the 4080S. THR takes about as long as P\*, and NC about as long as BGB. [internal note omitted]
- **Hard cap**: GPU ≤ 16 hours, wall clock ≤ 3 days. If exceeded, tiers are cut in the order C → B.

## Decision (written in advance)

The main decision uses only the COCO variants (tiers A and B). Define:
- Δ_voc = NC − BGB;
- Δ_thr = THR − P\*.

- **COLLISION_SUPPORTED**, when (a) and (b) both hold:
  - (a) the mean of Δ_voc over all methods and variants in tiers A and B ≥ +1.0;
  - (b) on obj_bg30, the number of methods with Δ_voc > 0 ≥ 5/7. For a method with zero collisions, Δ_voc is recorded as 0 and counts as "not greater than 0".
- **PARTIAL**: only one of (a) and (b) holds.
- **NOT_SUPPORTED**: neither (a) nor (b) holds.
- If tier B is cut, (a) is computed from tier A only. If tier A is incomplete, the conclusion is INSUFFICIENT.

**Descriptive readouts (not part of the decision):**
- Mean and sign of Δ_thr: whether the numerical part alone is harmful;
- NC − P\*: whether the gain turns positive once the colliding words are removed;
- Δ_voc and Δ_thr on tier C: in a setting where the background trick is positive, which part the gain comes from;
- The list of deleted colliding words for each method and each variant.

**How the outcome is used in the paper:**
- COLLISION_SUPPORTED: we may write "conflicts between the transplanted background words and the query class names are the main reason the official trick fails on a new class list", and use this as mechanism evidence in Section 4.4.
- PARTIAL or NOT_SUPPORTED: collision is kept only as a hypothesis in the discussion, and the effect of the numerical part is reported as it is.

## Implementation

- New script s2_run_one.py: obtained from S1's s1_run_one.py by adding the two settings THR and NC. The P\* and BGB code paths are unchanged.
- All S2 cells are run through S1's low-memory wrapper (the same per-class rewrite as in s1_run_one_lm.py, S1 deviation 6), to avoid OOM and subsequent reruns; this rewrite has already been verified to be bit-for-bit equivalent.
- **Implementation check (must pass before inference)**: using the BGB path of s2_run_one, rerun S1's ClearCLIP BGB obj_bg30 (S1 value 18.12) and SCLIP P ctx_bg70 (S1 value 20.08), each within ±0.05, on the same GPU model. If this fails, investigate; do not relax the threshold.
- Smoke-test THR and NC once on 20 images before the main run.

## Machine and data

- Use an RTX 4080 SUPER instance [internal note omitted]. S1's derived data, specs, names and code are all on it; no new data is needed.
- If this instance has no free GPU:
  - wait at most 24 hours;
  - after that, consult the author; do not migrate on our own.
- [internal note omitted]

## Time constraints

- [internal note omitted]
- S2 is planned to be completed within a few days of approval [internal note omitted].

## Rules

- Inference only. The collision rule, setting definitions and thresholds are all frozen before any S2 computation.
- No data, threshold, seed or criterion of S1 or of any earlier test is changed.
- No password, API key or token is entered or stored.

## Deviations (appended section)

1. **Switch to an RTX 3090 instance** (2026-11-08, about 21:30 Beijing time, before any S2 computation).
   - Reason: the RTX 4080 SUPER instance had no free GPU when it was started, whereas an RTX 3090 instance had GPUs available [internal note omitted]. Approved by the author on 2026-11-08.
   - The RTX 3090 instance already holds S1's derived data, specs and names. These were generated on it by S1 Tier 1, and the sha256 of the merge list, cc5d61b6…, is identical to that on the RTX 4080 SUPER instance.
   - s1_run_one_lm.py (md5 464580cb) and the three S2 scripts had to be added to this instance [internal note omitted].
   - The S2 cells are run on the 3090, while the controls are S1's P\* and BGB from the 4080S. The implementation-check threshold is unchanged: recompute the two S1 values on the RTX 3090 instance, ClearCLIP BGB obj_bg30 = 18.12 and SCLIP P ctx_bg70 = 20.08, each ±0.05. If the tolerance is exceeded, stop; do not relax it. In S1, the cross-GPU difference on ctx_bg0 was 0.01.
   - [internal note omitted]
2. **Collision list generated and frozen** (2026-11-08, about 21:40, before any S2 inference; this follows the rule in the body and is not a change to the rule).
   - Code files on the RTX 3090 instance [internal note omitted]: s1_run_one_lm.py (464580cb), mk_s2_names.py (1f3c2ade), s2_run_one.py (83851875), s2_stage.sh (b35315f0).
   - The sha256 of s2_collisions.json is c8ea1eb594d160afe13945ef478efb8d5460bfc10de3d8f75b502cc15a913f4b.
   - Result of applying the rule in the body:
     - obj_bg30: for SCLIP, NACLIP, ResCLIP and SC-CLIP, 15 of the 26 background words are deleted; for ProxyCLIP and Trident, 9 of the 17 are deleted.
     - obj_bg50: 13 and 8 are deleted, respectively.
     - ClearCLIP's official background line contains only "background", so it has zero collisions on both COCO variants. Per the body, these two cells are not run and Δ_voc is recorded as 0; consequently (b) can reach at most 6/7.
     - ctx_bg70: the official Context-60 background lines of all 7 methods contain only "background"; all have zero collisions, so none of the tier-C NC cells is run, and tier C actually runs only the 7 THR cells.
     - The deleted words include bed: it has the same name as the COCO object class bed, which shows that this collision already existed in the BGB of the original COCO-Object (descriptive observation).
   - Cells actually to be run: 13 in tier A, 13 in tier B, 7 in tier C, 33 in total, plus 2 check cells and 2 smoke-test cells.
   - [internal note omitted]
3. **The compute instance was stopped by the cloud provider, interrupting the run** (2026-11-09, discovered at a check at 12:40 Beijing time) [internal note omitted].
   - The instance stopped at about 11:08 [internal note omitted].
   - [internal note omitted]
   - The last check before the stop was at 08:34: 22/33 cells had been completed, with no failed cell. Progress after that was unknown and had to be read from stage_log.txt after the instance was restarted [internal note omitted].
   - Per the contract:
     - the interruption does not change any threshold;
     - unfinished cells are completed after resumption by restarting from the checkpoint (s2_stage.sh skips cells that already have results);
     - if some tier-A/B cells are not completed, they are handled according to the INSUFFICIENT and tier-cutting rules in the body;
     - the 16-hour GPU cap is counted cumulatively over the time used.
   - [internal note omitted]
4. **Progress read after restart; tier C not completed; scored by the main decision** (2026-11-09, about 14:10 Beijing time).
   - The RTX 3090 instance was started in GPU-less mode to read the log [internal note omitted]:
     - all 26 cells of tiers A and B completed, rc=0;
     - tier C THR completed 3/7 (SCLIP, ClearCLIP, NACLIP);
     - the stop occurred while ResCLIP THR ctx_bg70 was running; this cell has no result file.
   - The instance had no free GPU at that time. Tier C is used only for descriptive readouts and does not enter the main decision, so the remaining 4 cells were not rerun on a cloned instance.
   - The main decision depends only on tiers A and B, which are complete; the conclusion is valid.
   - The transcribed cell results were verified to be identical to the cell results on the instance [internal note omitted].
   - [internal note omitted]
