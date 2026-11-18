# ovss-collision-check

A small, dependency-free tool that checks whether a **background vocabulary** used by a training-free
open-vocabulary semantic segmentation (OVSS) method collides with the **target class names**.

Many training-free OVSS methods represent the background class by a list of words (for example
`sky, wall, tree, wood, grass, road, ..., bed, ...`). When one of those words also names a class you want to
segment, pixels of that class are pulled into background. On COCO-Object, for instance, six of the seven
official configurations we re-ran list `bed` as a background word although `bed` is a target class.
This tool finds such words before you run inference and writes a cleaned class-name file.

It implements exactly the rule described in Section 3.3 of the paper and used in its pre-registered
experiments.

## The rule

The check uses class-name strings only (no images, no predictions).

1. **Tokenise.** Split the background line and every query line by the separator (`;` for ProxyCLIP- and
   Trident-style name files, `,` otherwise) and strip whitespace.
2. **Normalise.** Lowercase; replace `-` and `_` by spaces; drop a trailing word `other` or `merged`;
   collapse whitespace.
3. **Match.** Background word *w* collides with query word *q* iff, after normalisation, *w* = *q*, or *w* is one
   of the space-separated words of *q*, or *q* is one of the space-separated words of *w*.
   Examples: `bed`/`bed`, `sky`/`sky-other`, `wall`/`wall-concrete`; but not `sea`/`seat`.
4. **Filter.** Remove colliding words from the background line only. Query lines are never changed and the
   word `background` is never removed. If nothing is left, the background line becomes `background`.

The rule is deliberately conservative: synonyms and part–whole relations (for example `rider` vs. `person`)
are **not** detected, so the losses it reveals may underestimate the full effect of a background vocabulary.

## Requirements

Python 3.8 or later, standard library only. Files are read and written as UTF-8.

## Usage

The input is a class-name file in the format used by SCLIP, ClearCLIP, NACLIP, ResCLIP, SC-CLIP, ProxyCLIP,
Trident and related repositories: one class per line, the first non-empty line is the background class, and a
line may hold several names separated by the separator.

```bash
# background line vs. the other lines of the same file
python ovss_collision_check.py cls_coco_object.txt

# background line vs. a different class list (e.g. a new domain), one class per line
python ovss_collision_check.py cls_coco_object.txt --targets my_classes.txt

# write the cleaned file and a JSON report; exit with status 1 if any collision is found (useful in CI)
python ovss_collision_check.py cls_coco_object.txt --out cls_coco_object_nc.txt --json report.json --strict

# force the separator instead of auto-detecting it
python ovss_collision_check.py cls_coco_object.txt --sep ';'
```

Example output:

```
background words: 17   colliding: 1   separator: ';'
                 'bed'  collides with  'bed'
clean background line: sky; wall; tree; wood; grass; road; sea; river; mountain; sands; desk; building; cloud; floor; hill; rail
```

`--sep auto` (default) chooses `;` if any line contains `;`, otherwise `,`. If your files mix both, set
`--sep` explicitly.

Exit status: `0` finished (also when collisions are found, unless `--strict`); `1` collisions found with
`--strict`; `2` error (missing or unreadable file, bad separator).

### Python API

```python
from ovss_collision_check import check
r = check("sky, wall, bed", ["person", "bed", "wall-concrete"], sep=",")
r["removed"]                # {'wall': ['wall-concrete'], 'bed': ['bed']}
r["clean_background_line"]  # 'sky'
```

## Files

| File | Purpose |
|---|---|
| `ovss_collision_check.py` | the tool (CLI and API) |
| `test_ovss_collision_check.py` | unit tests: `python -m unittest test_ovss_collision_check.py` |
| `regression_check.py` | re-derives the paper's collision lists from the released name files and official repositories (needs the experiment archives, which are not part of this folder) |
| `examples/` | two toy files used by the tests |

## Verification

`regression_check.py` rebuilds every background-handling (BGB) name file of the paper's experiments S2 and S3
(official background line + released query lines), applies the tool and compares the result with the released
collision-free (NC) name files and collision lists. All 35 cases (21 in S2: COCO-Stuff and Context variants;
14 in S3: original VOC21 and COCO-Object) are identical byte for byte.

## Citation and licence

MIT License (see the repository's `LICENSE`). If you use the tool, please cite the paper (citation details will be added on publication) or the repository via its `CITATION.cff`.
