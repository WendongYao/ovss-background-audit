# Contract V4: Probe V4 (first elimination gate): does the stage-3 conclusion still hold with a different, automatically generated background vocabulary?

> English translation of the analysis contract for stage V4, frozen on 2026-11-06 before any computation for that stage. Deviations were appended afterwards with their dates and are translated in full. The original is in Chinese. Internal operational notes have been removed and marked `[internal note omitted]`: compute-instance identifiers, billing, file transfers, and chat references. No question, setting, decision rule, threshold, seed, dataset, outcome or deviation was changed. Translated 2026-11-18.

Date: 2026-11-06. Approved by the author on 2026-11-06. Upstream: Probe_OVSSBackground03 (ADVANCE; [internal note omitted]). Inference only. The contract was written before the vocabulary was generated and before any inference; it is not changed after results are available, and deviations are appended at the end.

## Question to answer

The rule selected in stage 3 (temperature 100, class-presence filter r = 0.6, per-image Otsu) uses the 26 background words from the SCLIP repository. If these 26 words are replaced with a background vocabulary generated automatically with the LLM method (CC^L) of Wysoczańska et al. (TMLR 2025), without looking at any image data, does the rule still match each method's tuned official background trick?

## Automatic background vocabulary V_CC (generated, with its SHA-256 recorded, before any inference)

- LLM: Qwen2.5-7B-Instruct (open source, no access approval required; the original paper uses Mixtral-8x7B-Instruct 4-bit, which does not fit in 24 GB of GPU memory; recorded as a deviation), fp16, greedy decoding, max_new_tokens = 128, single-turn conversation.
- Prompt: the verbatim prompt from Appendix D.2, Figure 14 of the original paper (including the two examples, fork and child), with {q} filled in with the class name; wrapped as a single user message using the model's own chat template.
- Post-processing (as in the original paper; format normalization only): parse into a word list; lowercase, strip leading/trailing whitespace and quotes; deduplicate. No other filtering.
- Handling in the multi-class setting (as in the original paper): generate for each foreground class of the dataset (unified class names, as in stages 2 and 3) and take the union; using OpenAI CLIP ViT-B/16 text features (template "a photo of a {}.", normalized), compute the cosine similarity with any foreground class name of that dataset, and remove words with > 0.9 (β = 0.9 in the original paper).
- Final background vocabulary = "background" + the words above (one for VOC21 and one for COCO-Object), saved as `bg_words_cc.json`.

## Rule and implementation

- Rule: exactly the same as the rule selected in stage 3 (V1 structure: background class = maximum over all words in the vocabulary; T = 100; r = 0.6; Otsu); only the 26 words are replaced by V_CC. **No parameters are re-selected on the development set.**
- Implementation: the stage-3 wrapper layer; the only change is that the background vocabulary is configurable. To avoid doubling the GPU memory of the method's own post-processing when there are more query words, all background queries are merged into one before the original `postprocess_result` is called (the values are likewise ±10; the decision is unchanged), and the corresponding query_idx is used temporarily.
- Implementation check (before testing; must pass): with this new code and the background vocabulary set to the original 26 words, run SCLIP and Trident on the VOC21 validation set; they must reproduce the stage-3 values 57.61 and 66.05, respectively (each ±0.05). If this fails, debug the implementation and do not proceed to testing.

## Test and reading

- Test cells: 7 methods × {VOC21, COCO-Object} validation sets, 14 in total; P\* settings as in stages 2 and 3.
- Δ_trick(CC) = mean over the 14 cells of [V_CC rule − stage-2 BGB].

| Condition | Conclusion |
|---|---|
| Δ_trick(CC) ≥ −2.0 | PASS: the conclusion does not depend on those 26 words; proceed to the second gate (more datasets) |
| −4.0 ≤ Δ_trick(CC) < −2.0 | DISCUSS: partial dependence; discuss with the author |
| Δ_trick(CC) < −4.0 | STOP: the conclusion depends on those 26 specific words, and the method part does not hold; return to "audit only" or to topic selection; discuss with the author |

- Descriptive readouts (do not change the decision): per-cell differences from stage 3 (26 words); by dataset; the 4 methods that were not involved in selecting the rule; vocabulary size and examples.

## Budget and rules

- GPU ≤ 8 hours; wall clock ≤ 3 days. Single-process run.
- Smoke test on 20 images first.
- Do not change the vocabulary, rule parameters, thresholds or reference numbers to rescue the results; the vocabulary is not changed after it is generated.

## Pre-run deviations (2026-11-06, appended before vocabulary generation and before any inference)

1. Hardware: the hosts of both original instances had no free GPU. The author cloned a new instance from the instance holding all stage-3 files [internal note omitted]; its GPU is a vGPU-32GB (RTX 4080 SUPER 32 GB, the same model as in stage 2). The implementation-check targets 57.61 / 66.05 come from the RTX 3090 in stage 3; stage 2 measured a numerical difference of about 0.03 between this model and the 3090, so the ±0.05 tolerance still applies. If the tolerance is exceeded because of the GPU change, investigate as in stage-3 deviation 2 and record it truthfully. The code p4_run.py was generated from the same p3_run.py by the same transformation script; the MD5s of the three scripts match the local copies (8c4aabbb, 6d560aa8, 0bd8a169).

## Mid-run deviations (2026-11-06, appended before any test inference)

2. The first run of vocabulary generation raised an error when writing the JSON, because of the system's default ASCII encoding (the generation itself had completed: VOC 83 words, COCO 240 words). The two file writes were changed to UTF-8 and the generation was rerun; greedy decoding is reproducible, and the word counts and the first few examples of the second run were identical to those of the first. The vocabulary SHA-256 is recorded in vocab_sha256.txt.
3. The COCO vocabulary contains 2 Chinese words ("卫生纸", "马桶刷", from the generation output for the toilet class). Following the contract ("format normalization only, no other filtering, no changes after generation"), they were kept as is. So that each repository reads and writes the class-name files as UTF-8, the environment variable PYTHONUTF8=1 was set for all inference (method code unchanged). The implementation check involves only VOC and is not affected.
4. Vocabulary observation (no changes made): the LLM output contains several part words of foreground classes (e.g., feather, wheel, tire, engine), although the prompt asks not to generate parts; the original paper does not filter these either, so they are kept, per the contract.
