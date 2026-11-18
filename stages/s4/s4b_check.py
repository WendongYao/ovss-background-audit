"""S4 deviation 6 checks: (B) gate for the 5 settings x 2 bed images + intermediate-map consistency flags;
(R) 3090 re-run of all THR / NC cells vs frozen values, and vs the 4080 renders (deviation 4).
Usage: python s4b_check.py <project_root> <out_json>"""
import glob
import io
import json
import sys
import tarfile

import numpy as np

ROOT, OUT = sys.argv[1:3]
R = f"{ROOT}/experiments/S4/results"
B = f"{R}/render_b"


def frozen(setting, ds):
    if setting in ("THR", "NC"):
        tar, mem = "s3_results.tar.gz", f"s3/cells/SCLIP__{setting}__{ds}_orig_perimage.npz"
    else:
        tar, mem = "p2_perimage_npz.tar.gz", f"results/full/SCLIP__{setting}__{ds}_perimage.npz"
    with tarfile.open(f"{ROOT}/archives/{tar}") as t:
        return dict(np.load(io.BytesIO(t.extractfile(mem).read())))


def miou(i, u):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(100 * np.nanmean(np.where(u > 0, i / u, np.nan)))


res = {"B": [], "R": []}
for s in ("P", "THR", "BGB", "NC", "O"):
    v = "coco_object_orig" if s in ("THR", "NC") else "coco_object"
    new = np.load(f"{B}/cells/b/SCLIP__{s}__{v}_perimage.npz")
    old = frozen(s, "coco_object")
    paths = json.load(open(f"{B}/out/b/SCLIP__{s}__{v}__paths.json"))["images"]
    for r, i in enumerate((369, 4575)):
        a, b = miou(new["intersect"][r], new["union"][r]), miou(old["intersect"][i], old["union"][i])
        res["B"].append({"setting": s, "index": i, "miou_render": round(a, 3), "miou_frozen": round(b, 3),
                         "diff": round(a - b, 3), "label_identical": bool(np.array_equal(new["label"][r], old["label"][i])),
                         "pred_area_identical": bool(np.array_equal(new["pred"][r], old["pred"][i])),
                         "consistent": paths[r]["consistent"], "pass": abs(a - b) <= 0.5 and paths[r]["consistent"]})

sel = {f: json.load(open(f"{R}/{f}.json")) for f in ("selection", "selection_supp")}
idx = {"q1": sel["selection"]["Q1"], "sq1": sel["selection_supp"]["Q1"]}
jobs = [("q1", "THR", "voc21"), ("q1", "THR", "coco_object"), ("q2", "NC", "coco_object"),
        ("sq1", "THR", "voc21"), ("sq1", "THR", "coco_object"), ("sq2", "NC", "coco_object")]
for g, s, ds in jobs:
    ids = ([x["index"] for x in idx[g][ds]["selected"]] if g in idx else
           [x["index"] for x in sel["selection" if g == "q2" else "selection_supp"]["Q2"]["coco_object"]["selected"]])
    n3090 = np.load(f"{B}/cells/r3090/{g}/SCLIP__{s}__{ds}_orig_perimage.npz")
    n4080 = np.load(f"{R}/render/cells/{g}/SCLIP__{s}__{ds}_orig_perimage.npz")
    old = frozen(s, ds)
    for r, i in enumerate(ids):
        f = miou(old["intersect"][i], old["union"][i])
        a = miou(n3090["intersect"][r], n3090["union"][r])
        c = miou(n4080["intersect"][r], n4080["union"][r])
        res["R"].append({"group": g, "setting": s, "dataset": ds, "index": i, "frozen": round(f, 3),
                         "r3090": round(a, 3), "r4080": round(c, 3), "diff3090": round(a - f, 3), "diff4080": round(c - f, 3),
                         "identical3090": bool(np.array_equal(n3090["intersect"][r], old["intersect"][i]) and
                                               np.array_equal(n3090["union"][r], old["union"][i])),
                         "pass3090": abs(a - f) <= 0.5})
res["B_pass"] = all(x["pass"] for x in res["B"])
res["R_pass3090"] = all(x["pass3090"] for x in res["R"])
res["R_identical3090"] = sum(x["identical3090"] for x in res["R"])
res["R_n"] = len(res["R"])
json.dump(res, open(OUT, "w"), indent=1)
for x in res["B"]:
    print("B", x)
print("B_pass", res["B_pass"])
for x in res["R"]:
    if not x["identical3090"] or abs(x["diff4080"]) > 0.5:
        print("R", x)
print("R_pass3090", res["R_pass3090"], "identical3090", res["R_identical3090"], "/", res["R_n"])
print("max|diff3090|", max(abs(x["diff3090"]) for x in res["R"]), "max|diff4080|", max(abs(x["diff4080"]) for x in res["R"]))
