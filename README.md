# ovss-background-audit

Code and data for the paper **“What the Background Hides: Thresholds, Vocabularies and Collisions in Training-Free Open-Vocabulary Segmentation”** (manuscript under review; citation details will be added on publication).

The paper audits how training-free open-vocabulary semantic segmentation (OVSS) methods built on CLIP decide which pixels are *background*. It re-runs seven methods from their official code (SCLIP, ClearCLIP, NACLIP, ResCLIP, SC-CLIP, ProxyCLIP, Trident) and compares each method's official configuration with a strict unified protocol. It then decomposes the official background handling into numeric settings and class-name files, and studies collisions between background words and target class names. Every experimental stage was run under an analysis contract frozen before computation.

## Contents

| Path | What it is |
|---|---|
| `tools/collision_check/` | **Stand-alone collision checker** (Python ≥ 3.8, standard library only). It finds background words that collide with target class names and writes a cleaned class-name file. See its own README. |
| `stages/v2_p2/` | Official configuration (O), strict unified protocol (P\*), transplanted official background handling (BGB) and Otsu runs; environment setup scripts |
| `stages/v3/` | Tuning-free background rules selected on training splits (SEL26) |
| `stages/v4/` | LLM-generated background vocabulary (CC) |
| `stages/v5/` | Context-60 replication and conflict-filtered vocabulary (CF) |
| `stages/s1/` | Derived datasets with controlled background share |
| `stages/s2/`, `stages/s3/` | Decomposition into thresholds (THR) and class-name files, and collision-free vocabularies (NC) |
| `stages/s4/` | Qualitative examples (deterministic image selection from frozen per-image statistics, rendering, consistency checks, figures) |
| `stages/s5/` | Collision mechanism: structural-zeroing check, per-class decomposition, leave-one-word-out ablation of the LLM vocabulary |
| `analysis/` | Image-level paired bootstrap and transcription checks |
| `paper_scripts/` | Scripts that produce the paper's figures and tables from the result files |
| `data/` | Result tables, decision files, collision manifests, derived-dataset merge lists and upstream commits (see `data/README.md`) |

## Environment

PyTorch 2.1.2 (CUDA 12.1), mmcv 2.1.0, mmengine 0.10.1, mmsegmentation 1.2.2. See `stages/v2_p2/setup_p2*.sh`.

- **Official method repositories are not included.** The setup scripts clone them; the exact commits we used are listed in `data/upstream/repo_commits.txt`. Please respect each repository's own license.
- **Datasets and model weights are not included.** These are PASCAL VOC 2012, COCO-Stuff/COCO-Object, PASCAL Context and ADE20K, plus CLIP ViT-B/16, DINO and SAM ViT-B weights. Obtain them from their original sources under their own terms. The data-preparation scripts are `stages/v2_p2/p2_data.py` and `stages/v5/p5_ctxdata.py`.
- The stage scripts were written for a single cloud GPU instance with data under `/root/autodl-tmp/`. Adjust the path constants at the top of each script if you run them elsewhere.

## Quick start: collision check

```bash
cd tools/collision_check
python ovss_collision_check.py examples/toy_coco_object.txt
python -m unittest test_ovss_collision_check.py
```

## Pre-registered contracts and per-image statistics

The frozen analysis contracts (with dated deviations) and the per-image intersection/union statistics behind the bootstrap intervals will be released here before publication. Large archives will be attached as release assets.

## License

- **Code** (all `.py` / `.sh` files): MIT License, see [`LICENSE`](LICENSE).
- **Data, result files, contracts and figures** (everything under `data/` and other non-code files): [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), see [`data/LICENSE`](data/LICENSE).
- Third-party datasets, models and the official method repositories remain under their own licenses and terms; nothing from them is redistributed here.

## Citation

If you use this code or data, please cite the paper (details will be added on publication) or this repository via [`CITATION.cff`](CITATION.cff).
