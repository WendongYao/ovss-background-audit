"""S5 layers 1-2 (frozen data only). See experiments/CONTRACT_S5_CN.md.
Usage: python s5_l12.py <project_root> <out_json>"""
import io
import json
import sys
import tarfile

import numpy as np

ROOT, OUT = sys.argv[1:3]
sys.path.insert(0, f"{ROOT}/tools/collision_check")
from ovss_collision_check import normalise  # noqa: E402

A = f"{ROOT}/archives"
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
SEP = {"ProxyCLIP": ";", "Trident": ";"}
_tars = {}


def member(tar, name):
    if tar not in _tars:
        _tars[tar] = tarfile.open(f"{A}/{tar}")
    try:
        return _tars[tar].extractfile(name).read()
    except KeyError:
        return None


def npz(tar, name):
    b = member(tar, name)
    return None if b is None else dict(np.load(io.BytesIO(b)))


def lines_of(b):
    return [l for l in b.decode("utf-8").split("\n") if l.strip()]


def names(line, m):
    return [x.strip() for x in line.split(SEP.get(m, ",")) if x.strip()]


s2c = json.loads(member("s2_results_partial.tar.gz", "s2/s2_collisions.json"))
s3c = json.loads(member("s3_results.tar.gz", "s3/s3_collisions.json"))

CELLS = []  # (dataset, method, collisions, class-lines source, BGB npz, O npz, NC npz)
for v in ("obj_bg30", "obj_bg50", "ctx_bg70"):
    for m in M7:
        col = s2c[v][m]
        CELLS.append((v, m, col, ("s1_results.tar.gz", f"s1/names/{v}__BGB__{m}.txt"),
                      ("s1_results.tar.gz", f"s1/cells/{m}__BGB__{v}_perimage.npz"), None,
                      ("s2_results_partial.tar.gz", f"s2/cells/{m}__NC__{v}_perimage.npz")))
for m in M7:
    col = s3c["coco_object_orig"][m]
    CELLS.append(("coco_object", m, col, ("s3_results.tar.gz", f"s3/names/coco_object_orig__NC__{m}.txt"),
                  ("p2_perimage_npz.tar.gz", f"results/full/{m}__BGB__coco_object_perimage.npz"),
                  ("p2_perimage_npz.tar.gz", f"results/full/{m}__O__coco_object_perimage.npz"),
                  ("s3_results.tar.gz", f"s3/cells/{m}__NC__coco_object_orig_perimage.npz")))


def ds_iou(d):
    i, u = d["intersect"].sum(0), d["union"].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(u > 0, i / u, np.nan)


L1, L2 = [], []
for ds, m, col, nsrc, bsrc, osrc, csrc in CELLS:
    if col["zero"]:
        continue
    ln = lines_of(member(*nsrc))
    bgline = names(ln[0], m)
    cls_lines = {k: names(l, m) for k, l in enumerate(ln) if k >= 1}
    B = npz(*bsrc)
    O = npz(*osrc) if osrc else None
    N = npz(*csrc)
    classes = {}
    for w, qs in col["removed"].items():
        raw_w = [x for x in bgline if normalise(x) == w] or [w]
        for q in qs:
            for k, nm in cls_lines.items():
                if normalise(q) in [normalise(x) for x in nm]:
                    exact = len(nm) == 1 and nm[0] in raw_w + ([w] if nsrc[0] == "s3_results.tar.gz" else [])
                    cat = "E" if exact else "T"
                    if classes.get(k) != "E":
                        classes[k] = cat
                    L1.append({"dataset": ds, "method": m, "bg_word": w, "query": q, "class_idx": k,
                               "class_line": nm, "cat": cat,
                               "pred_px_BGB": int(B["pred"][:, k].sum()),
                               "pred_px_O": None if O is None else int(O["pred"][:, k].sum()),
                               "pred_px_NC": None if N is None else int(N["pred"][:, k].sum()),
                               "gt_px": int(B["label"][:, k].sum())})
    if N is not None:
        ib, inc = ds_iou(B), ds_iou(N)
        valid = ~np.isnan(ib) & ~np.isnan(inc)
        C = int(valid.sum())
        d = np.where(valid, inc - ib, 0.0)
        E = [k for k, c in classes.items() if c == "E"]
        T = [k for k, c in classes.items() if c == "T"]
        rest = [k for k in range(len(d)) if k not in classes]
        L2.append({"dataset": ds, "method": m, "C_valid": C, "dvoc": round(100 * float(d.sum()) / C, 3),
                   "contrib_E": round(100 * float(d[E].sum()) / C, 3), "contrib_T": round(100 * float(d[T].sum()) / C, 3),
                   "contrib_rest": round(100 * float(d[rest].sum()) / C, 3),
                   "E_classes": {str(k): {"line": cls_lines[k], "IoU_BGB": round(100 * float(ib[k]), 2),
                                          "IoU_NC": round(100 * float(inc[k]), 2)} for k in E}})

res = {"L1": L1, "L2": L2}
e = [x for x in L1 if x["cat"] == "E"]
res["P1_n_E"] = len(e)
res["P1_exceptions"] = [x for x in e if x["pred_px_BGB"] != 0]
res["P1_pass"] = len(res["P1_exceptions"]) == 0
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print("L1 rows", len(L1), "E rows", len(e), "P1_pass", res["P1_pass"], "exceptions", len(res["P1_exceptions"]))
for x in L1:
    print("L1", x["dataset"], x["method"], x["bg_word"], "->", x["query"], x["class_line"], x["cat"],
          "BGB", x["pred_px_BGB"], "O", x["pred_px_O"], "NC", x["pred_px_NC"], "gt", x["gt_px"])
for x in L2:
    print("L2", x["dataset"], x["method"], "dvoc", x["dvoc"], "E", x["contrib_E"], "T", x["contrib_T"], "rest", x["contrib_rest"],
          {k: (v["line"], v["IoU_BGB"], v["IoU_NC"]) for k, v in x["E_classes"].items()})
