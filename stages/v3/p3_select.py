"""V3 rule selection on the dev caches (frozen family of 120 rules). Usage: python p3_select.py CACHE_ROOT OUT_JSON
CACHE_ROOT/<METHOD>__<ds>/NNNNN.npz with x [Q,h,w] fp16 similarities and y [h,w] uint8 labels (255 = ignore)."""

import glob
import itertools
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/p3/code")
from p3_run import BG26, decide  # noqa: E402

NBG = 1 + len(BG26)
NCLS = {"voc21": 21, "coco_object": 81}
RULES = [dict(V=v, T=t, r=r, D=d) for v, t, r, d in itertools.product(
    [0, 1], [40, 70, 100], [0.0, 0.2, 0.4, 0.6], ["argmax", "otsu", "0.3", "0.5", "0.7"])]


def simplicity(rule):  # tie-break: fewer components first (CONTRACT)
    return (rule["r"] > 0, rule["V"], ["argmax", "otsu", "0.3", "0.5", "0.7"].index(rule["D"]) > 1,
            ["argmax", "otsu"].index(rule["D"]) if rule["D"] in ("argmax", "otsu") else 2, rule["T"] != 100)


def main(root, out):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = {}
    for d in sorted(glob.glob(f"{root}/*__*")):
        key = d.split("/")[-1]
        ds = key.split("__")[1]
        C = NCLS[ds]
        inter = torch.zeros(len(RULES), C, dtype=torch.float64, device=dev)
        union = torch.zeros(len(RULES), C, dtype=torch.float64, device=dev)
        files = sorted(glob.glob(f"{d}/*.npz"))
        for f in files:
            z = np.load(f)
            x = torch.from_numpy(z["x"]).to(dev).float()
            y = torch.from_numpy(z["y"].astype(np.int64)).to(dev)
            valid = y != 255
            for j, rule in enumerate(RULES):
                bg, keep = decide(x, NBG, rule, torch)
                pf = (rule["T"] * torch.cat([x[:1 if rule["V"] == 0 else NBG], x[NBG:]], 0)).softmax(0)
                pf = pf[1 if rule["V"] == 0 else NBG:]
                pf[~keep] = -1.0
                pred = pf.argmax(0) + 1
                pred[bg] = 0
                pv, yv = pred[valid], y[valid]
                inter[j] += torch.bincount(yv[pv == yv], minlength=C)[:C].double()
                area_p = torch.bincount(pv, minlength=C)[:C].double()
                area_y = torch.bincount(yv, minlength=C)[:C].double()
                union[j] += area_p + area_y - torch.bincount(yv[pv == yv], minlength=C)[:C].double()
        iou = (inter / union).cpu().numpy()
        miou = np.nanmean(np.where(union.cpu().numpy() > 0, iou, np.nan), axis=1) * 100
        scores[key] = {"n_images": len(files), "mIoU": miou.tolist()}
        print(key, len(files), "best %.2f" % miou.max(), "R0 %.2f" % miou[RULES.index(dict(V=0, T=100, r=0.0, D="otsu"))], flush=True)
    keys = sorted(scores)
    mean = np.mean([scores[k]["mIoU"] for k in keys], axis=0)
    best = mean.max()
    cands = [j for j in range(len(RULES)) if mean[j] >= best - 0.1]
    pick = min(cands, key=lambda j: (simplicity(RULES[j]), -mean[j]))
    res = {"selected_rule": RULES[pick], "selected_dev_mean": float(mean[pick]), "best_dev_mean": float(best),
           "tie_candidates": [RULES[j] for j in cands], "cells": keys, "rules": RULES, "dev_mean": mean.tolist(),
           "scores": scores, "R0_dev_mean": float(mean[RULES.index(dict(V=0, T=100, r=0.0, D="otsu"))])}
    json.dump(res, open(out, "w"), indent=1)
    json.dump(RULES[pick], open(out.replace(".json", "_rule.json"), "w"))
    print("SELECTED", RULES[pick], "dev mean %.2f (best %.2f, R0 %.2f)" % (mean[pick], best, res["R0_dev_mean"]))
    order = np.argsort(-mean)[:10]
    for j in order:
        print("top", RULES[j], "%.2f" % mean[j])


if __name__ == "__main__":
    main(*sys.argv[1:3])
