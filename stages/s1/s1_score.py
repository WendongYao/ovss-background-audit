"""S1 H1 scoring (contract experiments/CONTRACT_S1_CN.md). Input: cells.csv (method,variant,setting,mIoU) for new S1 cells.
Existing points (VOC21, COCO-Object from V2-P2 transcribed table; Context-60 from V5 RESULT Q1 table) are fixed below."""
import csv
import json
import sys

import numpy as np
from scipy.stats import spearmanr

M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
BG = {"voc21": 0.7331802130973488, "coco_object": 0.7048971365872848, "context60": 0.12574220612388162,
      "ctx_bg30": 0.3487678347884559, "ctx_bg50": 0.5295540296268324, "ctx_bg70": 0.7073749047237373,
      "obj_bg30": 0.3148891562399032, "obj_bg50": 0.5057103472634866}  # tier1_bgfrac.json
EXIST = {  # (P*, BGB) per method in M7 order
    "voc21": ([35.91, 37.77, 36.79, 36.66, 42.07, 40.57, 40.02], [60.76, 51.80, 57.02, 60.35, 64.34, 61.27, 67.08]),
    "coco_object": ([21.68, 23.32, 21.55, 22.00, 24.74, 23.89, 24.90], [33.07, 33.04, 32.34, 34.56, 36.75, 36.69, 41.29]),
    "context60": ([30.41, 31.45, 30.65, 31.84, 34.49, 34.48, 36.17], [31.93, 32.57, 31.89, 33.41, 36.33, 35.74, 38.03]),
}


def main(path, out):
    cells = {}
    for r in csv.DictReader(open(path)):
        cells[(r["method"], r["variant"], r["setting"])] = float(r["mIoU"])
    D = {k: np.array(b) - np.array(p) for k, (p, b) in EXIST.items()}
    for v in ["ctx_bg30", "ctx_bg50", "ctx_bg70", "obj_bg30", "obj_bg50"]:
        if all((m, v, s) in cells for m in M7 for s in ("P", "BGB")):
            D[v] = np.array([cells[(m, v, "BGB")] - cells[(m, v, "P")] for m in M7])
    pts = list(D)
    x = np.array([BG[p] for p in pts]); Y = np.stack([D[p] for p in pts])  # [points, 7]
    y = Y.mean(1)
    rec = {"points": {p: {"bg_frac": BG[p], "mean_dBG": float(D[p].mean()), "dBG": D[p].round(4).tolist()} for p in pts},
           "n_points": len(pts)}
    if len(pts) < 5:
        rec["H1"] = "INSUFFICIENT"
    else:
        rho = float(spearmanr(x, y).correlation)
        slope = float(np.polyfit(x, y, 1)[0])
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, 7, 7)
            bs.append(np.polyfit(x, Y[:, idx].mean(1), 1)[0])
        lo, hi = np.percentile(bs, [5, 95])
        seq = [p for p in ["context60", "ctx_bg30", "ctx_bg50", "ctx_bg70"] if p in D]
        mono = [bool(np.all(np.diff([D[p][i] for p in seq]) >= 0)) for i in range(7)]
        a = rho >= 0.8; b = slope > 0 and (lo > 0 or hi < 0) and lo > 0; c = len(seq) == 4 and sum(mono) >= 5
        rec.update({"spearman": rho, "slope": slope, "slope_boot90": [float(lo), float(hi)], "ctx_seq": seq,
                    "monotone_methods": dict(zip(M7, mono)), "n_monotone": int(sum(mono)), "a": bool(a), "b": bool(b), "c": bool(c)})
        rec["H1"] = "SUPPORTED" if (a and b and c) else ("PARTIAL" if (a or b) else "NOT_SUPPORTED")
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "points"}, indent=1))
    for p in pts:
        print(p, round(BG[p], 4), round(float(D[p].mean()), 2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
