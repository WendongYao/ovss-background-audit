# data/

Result and decision files used in the paper. License: CC BY 4.0 (see `LICENSE` in this folder).

| File | Content |
|---|---|
| `literature_audit.csv` | Configuration and paper audit of 13 training-free OVSS methods (Table 1 of the paper) |
| `upstream/repo_commits.txt` | Commits of the official repositories used in all runs |
| `derived_datasets/tier1_bgfrac_frozen.json` | Background shares and merge lists of the derived datasets (frozen before inference) |
| `collisions/s2_collisions.json`, `collisions/s3_collisions.json` | Background-word collision manifests (derived variants; original VOC21 and COCO-Object) |
| `results/v2p2_cells.csv` | mIoU under O, P\*, BGB and Otsu (seven methods × VOC21/VOC20/COCO-Object/ADE20K) |
| `results/s1_cells.csv`, `results/s1_h1_score.json` | Derived-dataset runs and the background-share analysis |
| `results/s2_cells.csv`, `results/s2_score.json` | THR / NC decomposition on derived variants |
| `results/s3_cells.csv`, `results/s3_score.json` | THR / NC decomposition on original VOC21 and COCO-Object |
| `results/v4_score.json` | LLM-generated vocabulary (CC) scores |
| `results/ci_bootstrap.json` | Image-level paired bootstrap intervals |
| `s4/selection*.json`, `s4/s4*_check.json` | Image selection for the qualitative figures and consistency checks |
| `s5/s5_l12.json`, `s5/A.json`, `s5/F.log` | Collision mechanism: structural-zeroing check, per-class decomposition, attribution set and recovery F |

Paths inside some JSON files refer to the original cloud instance (`/root/autodl-tmp/...`) and are kept for provenance.
