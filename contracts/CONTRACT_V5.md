# Contract V5: Probe V5 (second gate): does the protocol audit replicate on PASCAL Context-60? + a principled conflict-filtered background vocabulary

> English translation of the analysis contract for stage V5, frozen on 2026-11-06 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-06. Approved by the author on 2026-11-06 (the second gate and "conflict filtering + vocabulary" are merged into a single run) [internal note omitted]. Upstream: V2-P2 (audit), V3 (background rule), V4 (vocabulary replacement: PASS, but VOC21 −4.38, COCO-Object +0.98). Inference only. The contract was written before the data conversion and before any inference; it is not changed after results are available, and deviations are appended at the end.

## Questions to answer

- **Q1 (go/no-go; decides whether to commit to the benchmark paper):** Does the protocol effect seen in stage 2 on VOC21 / COCO-Object (a large gap between the official configurations and the fixed setting P\*, and method rankings that change with the setting) replicate on a new dataset with background, PASCAL Context-60?
- **Q2 (content; does not decide go/no-go):** After the automatically generated background vocabulary is cleaned with a conflict filter that depends only on class names and is written in advance (LLM judgment of "subtype / synonym / part"), can the rule match the official background trick on VOC21, COCO-Object and Context-60 alike?

## Data: PASCAL Context-60 validation set

- Images: the VOC2012 JPEGImages already contain all VOC2010 images. Validation list = images from 2010 or earlier in VOC2012 `ImageSets/Main/val.txt` that have Context annotations; 5,105 images are expected. If the count differs, record it and stop to investigate.
- Annotations: the official PASCAL Context trainval.tar.gz (Stanford homepage, one .mat per image, 459 classes), labels.txt, 59_labels.txt. The 59 classes are mapped to 1–59 in ascending order of their 459-class index; everything else (including unlabeled) becomes 0 (background), consistent with the mmseg PascalContext conversion convention. File SHA-256s are recorded.
- [internal note omitted]

## Methods and settings

The 7 methods are the same as in stages 2 and 3 (SCLIP, ClearCLIP, NACLIP, ResCLIP, SC-CLIP, ProxyCLIP, Trident); code, weights, the P\* definition, and the P\* rule for ResCLIP (stage-2 deviation 6/6a) are all unchanged. On Context-60, each method has 5 cells:

1. O: the official cfg_context60, unchanged;
2. P\*: the fixed setting (unified class names, short side 448, plain background, threshold 0, no post-processing);
3. BGB: P\* + transplanted official background trick;
4. SEL26: the rule selected in stage 3 (26 background words, T = 100, r = 0.6, Otsu);
5. CF: the same rule, with the background vocabulary replaced by the conflict-filtered vocabulary (see below).

In addition, CF is run for the 7 methods on each of the VOC21 and COCO-Object validation sets (14 cells). 49 cells in total.

## Conflict-filtered vocabulary CF (generated, with its SHA-256 recorded, before any inference)

- Starting point: the LLM-generated vocabulary of V4 (VOC21 and COCO-Object use V4's bg_words_cc.json directly); for Context-60, the vocabulary is generated for its 59 foreground classes with the same script, the same prompt and the same decoding settings, and passed through the same β = 0.9 CLIP filter.
- Filter 1: words containing non-ASCII characters are removed (the CLIP text encoder supports only English; such words were kept as is in V4, and this filter is written in advance here as part of CF).
- Filter 2 (LLM judgment): Qwen2.5-7B-Instruct, greedy decoding, one query per word, with the fixed prompt below; a word is removed if the answer begins with yes:

  "You are given a word W and a list of object categories. Answer yes if W names one of the categories, a synonym of one, a specific kind (subtype) of one, or a part of one. Otherwise answer no.
  W: {w}
  Categories: {foreground class names of the dataset, comma-separated}
  Answer with yes or no."
- No parameter is tuned on the development or test set; the filtered vocabulary is CF.

## Reproduction gate (data-conversion check)

The O values on Context-60 are compared with each method's self-reported numbers (recorded before running): SCLIP 31.74 (README), ClearCLIP 32.6 (paper Table 3), NACLIP 34.99 (README), ResCLIP (NACLIP base) 33.5 (paper Table 2), SC-CLIP 36.8 (paper Table I), ProxyCLIP 35.3 (paper Table 1, "Context" column), Trident 38.6 (paper Table 2). The data conversion is considered correct only if at least 4 of the 7 are within ±1.0; otherwise the result is CHECK_DATA: investigate the conversion, and do not read Q1 or Q2.

## Reading

**Q1 (Context-60)**

- (a) Mean difference between official configuration and P\*: mean(O − P\*) ≥ 5.0;
- (b) Ranking change: when the 7 methods are ranked by O and by P\*, the number of discordant method pairs is ≥ 3 / 21 (i.e., Kendall τ ≤ 0.71).

| (a) | (b) | Conclusion |
|---|---|---|
| holds | holds | GO: commit to the benchmark paper [internal note omitted]; write a full plan |
| holds | does not hold | GO_NARROW: audit only background handling; discuss the scope with the author |
| does not hold | — | STOP: the protocol effect does not generalize; stop and return to topic selection |

**Q2 (does not decide go/no-go)**

- Δ_trick(CF) = mean over 21 cells (3 datasets with background × 7 methods) of [CF − BGB]. BGB: stage-2 numbers for VOC21 and COCO-Object; numbers from this run for Context-60.
- If Δ_trick(CF) ≥ −2.0 and Δ_trick(CF) on VOC21 ≥ −2.0: the paper adopts CF as the "principled, tuning-free background baseline". Otherwise, "background-vocabulary sensitivity" is reported as a finding, the 26-word rule serves as a reference heuristic, and the method is not presented as a contribution.
- Descriptive readouts: per-cell differences of CF from the 26 words and from the V4 automatic vocabulary; changes in per-class IoU on VOC21 (to test the collision explanation); the list of words that were filtered out.

## Implementation check (must pass before testing)

- The extended run script reproduces on VOC21: SCLIP O = 59.63 (stage 2) and SCLIP SEL26 = 57.60 (V4, same GPU model), each ±0.05. If this fails, debug the implementation and do not proceed to testing.

## Budget and rules

- GPU ≤ 20 hours; wall clock ≤ 5 days. Single-process run. Smoke test on 20 images first.
- Data preparation is done in GPU-less mode as far as possible.
- Do not change the data, vocabulary, rule, thresholds or reference numbers to rescue the results; the vocabulary is not changed after it is generated.

## Deviations (appended, before any inference)

1. **Data source**: the originally planned Stanford trainval.tar.gz (.mat annotations) returned 404 and the codalab trainval_merged.json returned 409; neither could be obtained. Instead, we use the Context-60 val annotation file val.pth released by PyTorch-Encoding (hangzh S3, 913,687,510 bytes, ETag edb3cf170802955cdb78cb6d6e8a57dc-55; after download, the ETag was verified by multipart MD5). This is a conversion from the same source as the 59-class + background mapping commonly used by mmseg and in the methods' papers. The pickle is loaded with a restricted deserializer (only the global names needed to rebuild PIL / numpy / collections / copyreg / torch storages are allowed; everything else is rejected), and the global names actually encountered are written to ctxdata_report.json. Images are taken from the existing VOC2012 JPEGImages (the VOC2010 images are a subset of them). The conversion must satisfy: 5105 images, labels 0–59, sizes identical to the images; afterwards, G0 (≥ 4 of 7 within ±1.0) tests whether the conversion is correct.
2. **Conversion done in GPU mode**: GPU-less mode has a memory limit of 2 GB, and loading the pickle of about 0.9 GB risks running out of memory; the conversion is therefore done together with inference in GPU mode. This does not affect any number.
3. **Switch to an RTX 3090 instance** (2026-11-06 21:40, before any inference): the host of the cloned instance used for V4 persistently had no free GPU [internal note omitted], and loading val.pth in GPU-less mode was killed by OOM (memory.events oom_kill 2; no file was written). The RTX 3090 instance used for stage 2 and V3 [internal note omitted] had a free GPU, so models/, p4/, p5/ and p2/data/raw/pcontext/ were copied from the cloned instance to it [internal note omitted] and verified: val.pth md5 92636c23dca1b3f0ef8b1fba3a9e6b9e (identical to the copy on the cloned instance); p4/bg_words_cc.json sha256 54bcf905… (identical to V4); the md5s of p4_run.py 8c4aabbb, run_one.py e1ced23a and all p5 scripts match the code of this contract; selected_rule.json is unchanged. All V5 cells (including CF vocabulary generation, Context-60 CC generation, and the implementation check) are run on this RTX 3090 instance. The implementation-check threshold is unchanged: the reference 57.60 for SCLIP SEL26 comes from V4 (vGPU); V3 obtained 57.61 on the 3090 with the same rule and the same vocabulary. If the check on the 3090 exceeds ±0.05 because of fp16 numerical differences, we stop, per the contract, and do not relax the threshold.
4. **val.pth contains only 5104 images** (21:37, before any inference): the conversion finished; all 5104 images were written, with no missing image; labels are 0–59, and all 60 values occur; background accounts for 12.6%; the only global names encountered were PIL.Image.Image and _codecs.encode. The assertion at the end of p5_ctxdata.py requires 5105 images, so the script exited with an error. At that point val.txt and the 5104 PNGs had all been written; the conversion itself was complete, but has 1 image fewer than the commonly cited 5105. The official .mat / json annotations cannot be obtained, so the missing image cannot be filled in. The effect on mIoU is about 1/5105, far smaller than the ±1.0 tolerance of G0. Decision: continue with 5104 images. The conversion script is neither changed nor rerun; p5_go.sh sees that val.txt already exists and skips the conversion. Whether the conversion is correct is still judged by G0; the G0 threshold is unchanged.
5. **Bug in the p5_stage.sh script** (fixed at 22:32; no Context-60 cell had been produced yet): the functions one() / rule() used, within a single local statement, a variable that is assigned in that same statement; under set -u this raised "s: unbound variable", and the script exited at 21:43 before the first cell, leaving the GPU idle for about 48 minutes. The fix only splits the local statement into two lines; nothing else was changed (new md5 06addd15ecd0bf74a7d3ada6e0ecacd9; the old version is kept as p5_stage_v1_bug.sh). After the fix, everything was rerun from the start: the implementation check again gave 59.63 / 57.61, and the smoke-test results were identical to those of the first attempt.
