"""S4 implementation check (contract gate): per-image mIoU of every rendered prediction vs the frozen per-image npz.
Pass iff |diff| <= 0.5 mIoU points for every (setting, image).
Usage: python s4_check.py <project_root> <cells_dir> <out_json>"""
import io
import json
import sys
import tarfile

import numpy as np

ROOT, CELLS, OUT = sys.argv[1:4]
GROUPS = []  # (prefix, selection file): main figures and supplementary set (deviation 2)
for pre, fn in (("", "selection.json"), ("s", "selection_supp.json")):
    sel = json.load(open(f"{ROOT}/experiments/S4/results/{fn}"))
    GROUPS.append((pre, {ds: [s["index"] for s in v["selected"]] for ds, v in sel["Q1"].items()},
                   [s["index"] for s in sel["Q2"]["coco_object"]["selected"]]))


def frozen(setting, ds):
    if setting in ("THR", "NC"):
        tar, mem = "s3_results.tar.gz", f"s3/cells/SCLIP__{setting}__{ds}_orig_perimage.npz"
    else:
        tar, mem = "p2_perimage_npz.tar.gz", f"results/full/SCLIP__{setting}__{ds}_perimage.npz"
    with tarfile.open(f"{ROOT}/archives/{tar}") as t:
        return dict(np.load(io.BytesIO(t.extractfile(mem).read())))


def miou(inter, union):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * np.nanmean(np.where(union > 0, inter / union, np.nan), axis=-1)


jobs = []
for pre, q1, q2 in GROUPS:
    jobs += [(pre + "q1", s, ds, q1[ds]) for ds in q1 for s in ("O", "P", "BGB", "THR")]
    jobs += [(pre + "q2", s, "coco_object", q2) for s in ("BGB", "NC")]
rows, ok = [], True
for grp, s, ds, idx in jobs:
    v = ds + "_orig" if s in ("THR", "NC") else ds
    new = np.load(f"{CELLS}/{grp}/SCLIP__{s}__{v}_perimage.npz")
    old = frozen(s, ds)
    for r, i in enumerate(idx):
        a = float(miou(new["intersect"][r], new["union"][r]))
        b = float(miou(old["intersect"][i], old["union"][i]))
        same_label = bool(np.array_equal(new["label"][r], old["label"][i]))
        passed = abs(a - b) <= 0.5 and same_label
        ok &= passed
        rows.append({"group": grp, "setting": s, "dataset": ds, "index": i, "miou_render": round(a, 3),
                     "miou_frozen": round(b, 3), "diff": round(a - b, 3), "label_identical": same_label, "pass": passed})
json.dump({"pass": ok, "rows": rows}, open(OUT, "w"), indent=1)
for x in rows:
    print(x)
print("S4_CHECK", "PASS" if ok else "FAIL")
