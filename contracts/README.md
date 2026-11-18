# Pre-registered analysis contracts

Every experimental stage in the paper was run under an analysis contract that was frozen before any computation for that stage. Each contract fixes:

- the questions and settings;
- the datasets;
- the decision metrics and numerical gates;
- the pre-committed reading of each outcome.

Hardware changes, data substitutions and implementation findings were appended afterwards as dated deviations. Contracts were never edited after results became available. Appendix C of the paper summarizes the contracts and their outcomes.

| Stage | File | Topic | Outcome (see the contract and the paper) |
|---|---|---|---|
| V2 | [CONTRACT_V2.md](CONTRACT_V2.md) | Do protocol differences rival method differences? | ADVANCE_TO_DESIGN |
| V2-P2 | [CONTRACT_V2-P2.md](CONTRACT_V2-P2.md) | Progress over SCLIP under a fixed protocol; tuning-free background rule | DISCUSS_STOP |
| V3 | [CONTRACT_V3.md](CONTRACT_V3.md) | Background rule selected on training splits (SEL26) | ADVANCE |
| V4 | [CONTRACT_V4.md](CONTRACT_V4.md) | Robustness to an LLM-generated background vocabulary (CC) | PASS |
| V5 | [CONTRACT_V5.md](CONTRACT_V5.md) | Replication on Context-60; conflict-filtered vocabulary (CF) | Q1 STOP |
| S1 | [CONTRACT_S1.md](CONTRACT_S1.md) | Protocol gap vs. background share (derived datasets) | H1 SUPPORTED, H2 NOT_SUPPORTED |
| S2 | [CONTRACT_S2.md](CONTRACT_S2.md) | Thresholds vs. background vocabulary on derived COCO variants | COLLISION_SUPPORTED |
| S3 | [CONTRACT_S3.md](CONTRACT_S3.md) | Thresholds vs. class-name files on the original datasets | Descriptive readouts |
| S4 | [CONTRACT_S4.md](CONTRACT_S4.md) | Qualitative examples (deterministic image selection) | No quantitative claim |
| S5 | [CONTRACT_S5.md](CONTRACT_S5.md) | Mechanism of collision losses | Descriptive follow-up |

## About these files

- **Translations.** The contracts were written in Chinese. These are English translations made on 2026-11-18. They do not change any question, setting, decision rule, threshold, seed, dataset, outcome or deviation. Wording that is inconsistent in the original was translated literally and not corrected.
- **Removed internal notes.** Internal operational notes were removed and marked `[internal note omitted]`. These were compute-instance identifiers, billing, file transfers between machines, and references to private correspondence. Approval quotes from private correspondence were replaced by "Approved by the author on <date>".
- **Hashes.** Hashes that identify the code, vocabularies, selection files, collision lists and merge lists were kept.
- **File paths.** Paths such as `experiments/S4/...`, `archives/...` or `paper/figures/...` refer to the authors' working directory.
  - The corresponding released code is in [`../stages/`](../stages/).
  - The released result tables, decision files and manifests are in [`../data/`](../data/).
  - The per-stage run outputs, including the per-image statistics, are on Zenodo: https://doi.org/10.5281/zenodo.24338613.
- **Stage names.** "Stage 1" and "stage 2" in V2–V3 refer to V2 and V2-P2.
