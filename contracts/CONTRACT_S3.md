# Contract S3: Supplementary experiment S3: Decomposing the official background trick (threshold / vocabulary) on the original VOC21 and COCO-Object

> English translation of the analysis contract for stage S3, frozen on 2026-11-09 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-09. Approved by the author on 2026-11-09.
The body was frozen before any S3 computation; nothing is changed after results come out, and deviations are appended at the end.
Upstream: V2-P2 (O, P\*, BGB), S2 (decomposition method and collision rule). S3 does not change any earlier result, threshold or decision.

## Question

On the original VOC21 and COCO-Object, BGB − P\* is large (21.83 and 12.24). How much of it comes from the numerical part (prob_thd, logit_scale, area_thd), and how much from the official class-name files (background synonyms and rewritten class names)? Is there a collision loss on the original datasets?

## Settings

The definitions are the same as in S2, except that the data are replaced by the original validation sets:
- **THR**: P\* plus the numerical part of the dataset's official configuration; the class-name file is the unified class names (P\*).
- **NC**: BGB with the collision words removed from the background row; otherwise identical to BGB. The collision rule is verbatim the same as in the S2 contract: word splitting, normalization, matching criterion, deletion from the background row only, and "background" is never deleted.
- Controls: the P\*, BGB and O of V2-P2, not rerun.

## Cells

| Dataset | Setting | Cells |
|---|---|---|
| VOC21 (1449 images) | THR × 7; NC × methods with collisions | ≤ 14 |
| COCO-Object (5000 images) | THR × 7; NC × methods with collisions | ≤ 14 |

Methods with zero collisions do not run NC; their BGB value is taken, and Δ_voc = 0 is recorded.
Budget: GPU ≤ 8 hours (the RTX 3090 instance used for V5 [internal note omitted]), wall clock ≤ 2 days. If exceeded, the COCO-Object NC cells are cut first.

## Readout (written in advance; S3 makes no go/stop decision, it only determines how the paper is written)

For each dataset, compute:
- Δ_BG = BGB − P\* (already available);
- Δ_thr = THR − P\*;
- Δ_name = BGB − THR (the net effect of the official class-name files);
- Δ_voc = NC − BGB.

All are averaged over the 7 methods. Define share_thr = mean Δ_thr / mean Δ_BG.

- share_thr ≥ 0.67: the paper writes "on the original datasets, the gain comes **mainly from the threshold**";
- share_thr ≤ 0.33: it writes "**mainly from the class names or the background vocabulary**";
- in between: it writes "**both matter**", and reports them separately.

Collisions:
- If on COCO-Object mean Δ_voc ≥ +0.5, and Δ_voc > 0 for at least 4 of the methods with collisions, the paper writes "the official trick on the original COCO-Object also has a measurable collision loss";
- otherwise it writes "on the original datasets, the collision loss is small or absent".
- VOC21 is computed in the same way, for description only.

## Implementation

- New script mk_s3.py: generates the specs for voc21_orig and coco_object_orig, and, following the S2 rules, generates the NC class-name files and the collision list s3_collisions.json (sha256 recorded).
  - Spec contents: the parent configs are cfg_voc21 / cfg_coco_object; the data root is the same as in V2-P2; the class names are the unified class names; the P\* list is the unified class-name file; the BGB list is each method's official class-name file.
- New script s3_run_one.py: obtained from s2_run_one.py by changing paths; the setting definitions are identical. Except for Trident, all methods use the low-memory per-class rewrite from S1 (verified to be bitwise equivalent).
- **Implementation check (must pass before inference)**: on the RTX 3090 instance, recompute three V2-P2 values through the s3 path, each within ±0.05:
  - ClearCLIP BGB voc21 = 51.80;
  - SCLIP P voc21 = 35.91;
  - ClearCLIP BGB coco_object = 33.04.

  If the check fails, stop and investigate; do not relax. Note that V2-P2 ran on a vGPU; V5 reproduced SCLIP O voc21 = 59.63 on the 3090, consistent with V2-P2.
- First run one smoke test with 20 images.

## Rules

- Inference only. The collision rule, the setting definitions and the readout thresholds are frozen before any S3 computation.
- No data, threshold or decision of V2–S2 is changed.
- No password, API key or token is entered or stored.

## Deviations (appended)

1. **Collision list (generated according to the rule in the body, before any S3 inference) and run arrangements** (2026-11-09).
   - Scripts: mk_s3.py (9f1a6a9d), s3_run_one.py (2309fbd0), s3_stage.sh (2beb00e4).
   - The sha256 of s3_collisions.json is afe80667….
   - Collisions obtained according to the rule in the body:
     - **VOC21: all 7 methods have zero collisions**; not a single NC cell is run, and Δ_voc is recorded as 0 for all of them. The 26 background words of the SCLIP family have no string conflict with the 20 VOC class names.
     - **COCO-Object: except for ClearCLIP (whose background row contains only "background"), each of the 6 methods has 1 collision word, namely "bed"** (the same name as the COCO object class bed).
   - Cells actually to be run: VOC21 THR 7 cells, COCO-Object THR 7 cells and NC 6 cells, 20 cells in total, plus 3 check cells and 2 smoke-test cells.
   - A watchdog script deadline.sh was set up on the machine: it automatically stops the remaining cells 8 hours after the start of the run.
   - [internal note omitted]
2. **Run completed** (2026-11-09 21:01, Beijing time).
   - All 20 cells finished with rc=0; the deadline was not triggered.
   - The md5 of cells.csv is 685b909a… [internal note omitted].
   - [internal note omitted]
   - Scoring: code/s3_score.py → results/s3_score.json.
