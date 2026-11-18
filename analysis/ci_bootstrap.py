"""Descriptive image-level paired bootstrap CIs for protocol gaps (no decision gates).
mIoU per resample = mean over classes with union > 0 of sum(intersect)/sum(union), x100 (as mmseg).
Paired: the same image resample is used for every method and setting of a dataset. 2000 resamples, seed 0.
Inputs: per-image npz from archives (V2-P2, V5, S1, S2), extracted to WORK."""
import glob, json, os, sys
import numpy as np

WORK = sys.argv[1]; OUT = sys.argv[2]
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
B = 2000


def load(path):
    d = np.load(path)
    return d["intersect"].astype(np.float64), d["union"].astype(np.float64)


def miou_w(W, I, U):  # W: [B, N] weights; I, U: [N, C]
    si, su = W @ I, W @ U
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(su > 0, si / su, np.nan)
    return np.nanmean(r, 1) * 100


def miou(I, U):
    su = U.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.where(su > 0, I.sum(0) / su, np.nan)) * 100)


def gap_ci(files_a, files_b, seed):
    """files_*: {method: npz path}; returns point and 90/95% CI of mean over methods of (a - b), and per method."""
    I0, _ = load(next(iter(files_a.values())))
    n = I0.shape[0]
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.ones(n) / n, size=B).astype(np.float64)
    per, boots = {}, []
    for m in M7:
        Ia, Ua = load(files_a[m]); Ib, Ub = load(files_b[m])
        assert Ia.shape[0] == Ib.shape[0] == n
        pt = miou(Ia, Ua) - miou(Ib, Ub)
        bs = miou_w(W, Ia, Ua) - miou_w(W, Ib, Ub)
        per[m] = {"point": round(pt, 2), "ci95": [round(float(x), 2) for x in np.percentile(bs, [2.5, 97.5])]}
        boots.append(bs)
    mb = np.mean(boots, 0)
    mp = float(np.mean([per[m]["point"] for m in M7]))
    return {"n_images": n, "mean_point": round(mp, 2), "mean_ci90": [round(float(x), 2) for x in np.percentile(mb, [5, 95])],
            "mean_ci95": [round(float(x), 2) for x in np.percentile(mb, [2.5, 97.5])], "per_method": per}


P2 = f"{WORK}/p2_perimage_npz/results/full"; P5 = f"{WORK}/p5_results/p5/cells"
S1 = f"{WORK}/s1_results/s1/cells"; S2 = f"{WORK}/s2_results/s2/cells"
f = lambda d, s, ds: {m: f"{d}/{m}__{s}__{ds}_perimage.npz" for m in M7}
jobs = {
    "O-P | voc21": (f(P2, "O", "voc21"), f(P2, "P", "voc21")),
    "O-P | voc20": (f(P2, "O", "voc20"), f(P2, "P", "voc20")),
    "O-P | coco_object": (f(P2, "O", "coco_object"), f(P2, "P", "coco_object")),
    "O-P | ade20k": (f(P2, "O", "ade20k"), f(P2, "P", "ade20k")),
    "O-P | context60": (f(P5, "O", "context60"), f(P5, "P", "context60")),
    "BGB-P | voc21": (f(P2, "BGB", "voc21"), f(P2, "P", "voc21")),
    "BGB-P | coco_object": (f(P2, "BGB", "coco_object"), f(P2, "P", "coco_object")),
    "BGB-P | context60": (f(P5, "BGB", "context60"), f(P5, "P", "context60")),
}
for v in ["ctx_bg30", "ctx_bg50", "ctx_bg70", "obj_bg30", "obj_bg50"]:
    jobs[f"BGB-P | {v}"] = (f(S1, "BGB", v), f(S1, "P", v))
res = {}
for i, (k, (a, b)) in enumerate(jobs.items()):
    if not all(os.path.exists(p) for p in list(a.values()) + list(b.values())):
        res[k] = "missing files"; print(k, "missing"); continue
    res[k] = gap_ci(a, b, seed=i)
    print(k, res[k]["n_images"], res[k]["mean_point"], res[k]["mean_ci95"])
# S2: NC - BGB (methods with collisions only; ClearCLIP excluded = 0 by construction)
for i, v in enumerate(["obj_bg30", "obj_bg50"]):
    ms = [m for m in M7 if m != "ClearCLIP"]
    M7b = M7
    globals()["M7"] = ms
    a = {m: f"{S2}/{m}__NC__{v}_perimage.npz" for m in ms}; b = {m: f"{S1}/{m}__BGB__{v}_perimage.npz" for m in ms}
    if all(os.path.exists(p) for p in list(a.values()) + list(b.values())):
        r = gap_ci(a, b, seed=100 + i); r["note"] = "mean over 6 methods with collisions (ClearCLIP has none)"
        res[f"NC-BGB | {v}"] = r; print("NC-BGB", v, r["mean_point"], r["mean_ci95"])
    globals()["M7"] = M7b
json.dump(res, open(OUT, "w"), indent=1)
