"""Apply the frozen V2-P2 readouts. Usage: python p2_score.py <results_dir> <out_json>"""

import glob
import json
import os
import sys

import numpy as np

DATASETS = ["voc21", "voc20", "coco_object", "ade20k"]
BG = ["voc21", "coco_object"]
NEW = ["ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
OLD = ["SCLIP", "ClearCLIP", "NACLIP"]
CORE = OLD + NEW
REPORTED = {  # paper ViT-B/16 numbers frozen in CONTRACT_CN.md (VOC21, VOC20, Object, ADE20K)
    "SCLIP": [59.1, 80.4, 30.5, 16.1], "ClearCLIP": [51.8, 80.9, 33.0, 16.7], "NACLIP": [58.9, 79.7, 33.2, 17.4],
    "ResCLIP": [61.1, 86.0, 35.0, 18.0], "SC-CLIP": [64.6, 84.3, 37.7, 20.1],
    "ProxyCLIP": [61.3, 80.3, 37.5, 20.2], "Trident": [67.1, 84.5, 41.1, 21.9]}
O_TARGET = dict(REPORTED, SCLIP=[59.63, 81.54, 33.52, 16.45], NACLIP=[64.07, 83.03, 36.19, 19.05])  # official-config targets
# (old methods: the phase-1 gate targets = repo README numbers for SCLIP/NACLIP, paper for ClearCLIP; new: paper rows)
LATER_CLIP = ["ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP"]
VFM = ["ProxyCLIP", "Trident"]


def miou(inter, union):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(inter.sum(0) / union.sum(0)) * 100)


def main(rdir, out):
    R, PI = {}, {}
    for f in glob.glob(f"{rdir}/*__*__*.json"):
        r = json.load(open(f))
        R[(r["method"], r["setting"], r["dataset"])] = r["mIoU"]
        npz = f.replace(".json", "_perimage.npz")
        if os.path.exists(npz):
            PI[(r["method"], r["setting"], r["dataset"])] = np.load(npz)
    res = {"cells_found": len(R)}
    # G0 for new methods (and cross-check for old ones)
    gate = {}
    for m in CORE:
        diffs = {d: (R[(m, "O", d)] - O_TARGET[m][i]) if (m, "O", d) in R else None for i, d in enumerate(DATASETS)}
        n_ok = sum(1 for v in diffs.values() if v is not None and abs(v) <= 1.0)
        gate[m] = {"diffs": diffs, "pass": n_ok >= 3}
    res["gate"] = gate
    new_fail = [m for m in NEW if not gate[m]["pass"]]
    res["new_methods_failing_gate"] = new_fail
    if len(new_fail) >= 2:
        res["status"] = "CHECK_IMPLEMENTATION"
    later = [m for m in LATER_CLIP if m in OLD or gate[m]["pass"]]
    # T1
    g_rep = float(np.mean([max(REPORTED[m][i] for m in later) - REPORTED["SCLIP"][i] for i in range(4)]))
    g_fix = float(np.mean([max(R[(m, "P", d)] for m in later) - R[("SCLIP", "P", d)] for d in DATASETS]))
    rng = np.random.default_rng(int.from_bytes(b"R3P2", "big"))
    boots = []
    for _ in range(2000):
        vals = []
        for d in DATASETS:
            n = PI[("SCLIP", "P", d)]["intersect"].shape[0]
            idx = rng.integers(0, n, n)
            s = miou(PI[("SCLIP", "P", d)]["intersect"][idx], PI[("SCLIP", "P", d)]["union"][idx])
            best = max(miou(PI[(m, "P", d)]["intersect"][idx], PI[(m, "P", d)]["union"][idx]) for m in later)
            vals.append(best - s)
        boots.append(np.mean(vals))
    p90 = float(np.percentile(boots, 90))
    T1 = g_fix <= 0.5 * g_rep and p90 <= g_rep
    res["T1"] = {"later_methods": later, "G_rep": g_rep, "G_fix": g_fix, "G_fix_p90": p90, "holds": bool(T1),
                 "per_dataset_fix": {d: max(R[(m, "P", d)] for m in later) - R[("SCLIP", "P", d)] for d in DATASETS}}
    vfm = [m for m in VFM if gate[m]["pass"]]
    if vfm:
        res["T1_vfm_descriptive"] = {
            "G_rep": float(np.mean([max(REPORTED[m][i] for m in vfm) - REPORTED["SCLIP"][i] for i in range(4)])),
            "G_fix": float(np.mean([max(R[(m, "P", d)] for m in vfm) - R[("SCLIP", "P", d)] for d in DATASETS]))}
    # T2
    rows = []
    for m in CORE:
        for d in BG:
            rows.append({"method": m, "dataset": d, "P": R[(m, "P", d)], "BGB": R[(m, "BGB", d)], "OTSU": R[(m, "OTSU", d)]})
    vs_trick = float(np.mean([r["OTSU"] - r["BGB"] for r in rows]))
    vs_plain = float(np.mean([r["OTSU"] - r["P"] for r in rows]))
    T2 = vs_trick >= -1.0 and vs_plain >= 3.0
    res["T2"] = {"otsu_minus_official_trick": vs_trick, "otsu_minus_plain": vs_plain, "holds": bool(T2), "rows": rows}
    if "status" not in res:
        res["status"] = {(True, True): "ADVANCE_FULL", (True, False): "ADVANCE_BENCHMARK",
                         (False, True): "DISCUSS", (False, False): "DISCUSS_STOP"}[(bool(T1), bool(T2))]
    res["table"] = {f"{m}|{s}|{d}": v for (m, s, d), v in sorted(R.items())}
    res["METHOD_GO"] = False
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("status", "gate", "T1", "T2") if k in res}, indent=1)[:4000])


if __name__ == "__main__":
    main(*sys.argv[1:3])
