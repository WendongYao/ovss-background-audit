"""S4 image selection: deterministic, from frozen per-image statistics only (no images viewed).

Usage: python select_s4.py <project_root> <out_json>
Rule: see experiments/CONTRACT_S4_CN.md (section "选图规则").
"""
import io
import json
import sys
import tarfile

import numpy as np

ROOT = sys.argv[1]
OUT = sys.argv[2]
QS = [10, 50, 90]
BED = 60


def load(tar, member):
    with tarfile.open(f"{ROOT}/archives/{tar}") as t:
        d = np.load(io.BytesIO(t.extractfile(member).read()))
        return {k: d[k].astype(np.float64) for k in d.files}


def p2(setting, ds):
    return load("p2_perimage_npz.tar.gz", f"results/full/SCLIP__{setting}__{ds}_perimage.npz")


def s3(setting, ds):
    return load("s3_results.tar.gz", f"s3/cells/SCLIP__{setting}__{ds}_orig_perimage.npz")


def miou(d):
    u = d["union"]
    iou = np.where(u > 0, d["intersect"] / np.maximum(u, 1), np.nan)
    return np.nanmean(iou, axis=1)


def pick(values, eligible):
    idx = np.flatnonzero(eligible)
    v = values[idx]
    out, used = [], set()
    for q in QS:
        qv = float(np.percentile(v, q))
        order = sorted(range(len(idx)), key=lambda j: (abs(v[j] - qv), idx[j]))
        j = next(j for j in order if int(idx[j]) not in used)
        used.add(int(idx[j]))
        out.append({"percentile": q, "q_value": round(qv, 4),
                    "index": int(idx[j]), "value": round(float(v[j]), 4)})
    return out, int(len(idx))


res = {"method": "SCLIP", "rule": "CONTRACT_S4_CN.md", "Q1": {}, "Q2": {}}

for ds in ["voc21", "coco_object"]:
    P, B, O = p2("P", ds), p2("BGB", ds), p2("O", ds)
    T = s3("THR", ds)
    n = P["label"].shape[0]
    assert all(x["label"].shape[0] == n for x in (B, O, T))
    assert np.array_equal(P["label"], B["label"]) and np.array_equal(P["label"], T["label"])
    lab = P["label"]
    bgshare = lab[:, 0] / np.maximum(lab.sum(1), 1)
    elig = (lab[:, 1:].sum(1) > 0) & (bgshare >= 0.2) & (bgshare <= 0.8)
    m = {k: miou(v) for k, v in {"P": P, "THR": T, "BGB": B, "O": O}.items()}
    d = m["BGB"] - m["P"]
    sel, ne = pick(d, elig)
    for s in sel:
        i = s["index"]
        s["bg_share"] = round(float(bgshare[i]), 4)
        s["miou"] = {k: round(100 * float(v[i]), 2) for k, v in m.items()}
    res["Q1"][ds] = {"n_images": n, "n_eligible": ne, "selected": sel}

B, N = p2("BGB", "coco_object"), s3("NC", "coco_object")
assert np.array_equal(B["label"], N["label"])
lab = B["label"]
bedshare = lab[:, BED] / np.maximum(lab.sum(1), 1)
elig = bedshare >= 0.05
iou_b = B["intersect"][:, BED] / np.maximum(B["union"][:, BED], 1)
iou_n = N["intersect"][:, BED] / np.maximum(N["union"][:, BED], 1)
sel, ne = pick(iou_n - iou_b, elig)
for s in sel:
    i = s["index"]
    s["bed_share"] = round(float(bedshare[i]), 4)
    s["bed_iou"] = {"BGB": round(100 * float(iou_b[i]), 2), "NC": round(100 * float(iou_n[i]), 2)}
res["Q2"]["coco_object"] = {"n_images": int(lab.shape[0]), "n_eligible": ne, "selected": sel}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
