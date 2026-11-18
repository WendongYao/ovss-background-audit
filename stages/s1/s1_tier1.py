"""S1 Tier 1 (CPU only): background pixel fractions, frozen merge lists for derived datasets, derived label maps,
and the H2 per-image analysis from existing per-image npz files. Contract: experiments/CONTRACT_S1_CN.md."""
import json
import os
import sys

import numpy as np
from PIL import Image

A = "/root/autodl-tmp"
S1 = f"{A}/s1"
VOC = f"{A}/p2/data/VOCdevkit/VOC2012"
CTX = f"{A}/p2/data/VOCdevkit/VOC2010"
COCOO = f"{A}/p2/data/coco_object"
STUFF = f"{A}/p2/data/stuffthingmaps/val2017"
U = f"{A}/p2/repos/ClearCLIP/configs"
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
THING = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33,
         34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
         62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89]  # p2_data.py
STUFF_IDS = list(range(91, 182))  # COCO-Stuff stuff ids 91..181 -> lines 81..171 of cls_coco_stuff.txt


def lines(p):
    return [l.strip() for l in open(p).read().split("\n") if l.strip()]


def hist_dir(files, nbins=256):
    h = np.zeros(nbins, np.int64)
    for f in files:
        h += np.bincount(np.array(Image.open(f)).ravel(), minlength=nbins)[:nbins]
    return h


def bgfrac(h):  # background = 0, ignore = 255
    return float(h[0] / (h[:255].sum()))


def first_reaching(order, h, valid_total, bg0, target):
    merged, bg = [], bg0
    for c in order:
        if bg / valid_total >= target:
            break
        merged.append(int(c))
        bg += h[c]
    return merged, float(bg / valid_total)


def stage_a():
    os.makedirs(S1, exist_ok=True)
    rep = {}
    ids = open(f"{VOC}/ImageSets/Segmentation/val.txt").read().split()
    h = hist_dir([f"{VOC}/SegmentationClass/{i}.png" for i in ids]); rep["voc21"] = {"bg_frac": bgfrac(h), "n": len(ids)}
    cids = open(f"{CTX}/ImageSets/SegmentationContext/val.txt").read().split()
    hc = hist_dir([f"{CTX}/SegmentationClassContext/{i}.png" for i in cids]); rep["context60"] = {"bg_frac": bgfrac(hc), "n": len(cids)}
    fo = sorted(os.listdir(f"{COCOO}/annotations/val2017"))
    ho = hist_dir([f"{COCOO}/annotations/val2017/{f}" for f in fo]); rep["coco_object"] = {"bg_frac": bgfrac(ho), "n": len(fo)}
    fs = sorted(os.listdir(STUFF))
    hs = hist_dir([f"{STUFF}/{f}" for f in fs])
    rep["coco_stuff_hist_check"] = {"n": len(fs), "pixels_total": int(hs.sum()), "unlabeled_255": int(hs[255])}
    assert abs(rep["context60"]["bg_frac"] - 0.126) <= 0.001, ("context60 bg_frac check failed", rep["context60"])
    # Ctx-bgK merge lists: all 59 classes by descending val pixel count (ties: class id)
    ctx_order = sorted(range(1, 60), key=lambda c: (-int(hc[c]), c))
    tot = hc[:255].sum()
    rep["ctx_variants"] = {}
    for t in (0.30, 0.50, 0.70):
        m, f = first_reaching(ctx_order, hc, tot, hc[0], t)
        rep["ctx_variants"][f"ctx_bg{int(t*100)}"] = {"merged": m, "bg_frac": f}
    rep["ctx_variants"]["ctx_bg0"] = {"merged": [], "bg_frac": rep["context60"]["bg_frac"]}
    # Obj variants: stuff ids by descending pixel count; background = stuff merged + 255 (as in COCO-Object)
    tots = hs.sum()
    bg0 = hs[255]
    st_order = sorted(STUFF_IDS, key=lambda c: (-int(hs[c]), c))
    fo_bg = rep["coco_object"]["bg_frac"]
    targets = [0.30] if fo_bg < 0.40 else [0.30, (0.30 + fo_bg) / 2]
    rep["obj_variants"] = {}
    for t in targets:
        m, f = first_reaching(st_order, hs, tots, bg0, t)
        rep["obj_variants"][f"obj_bg{int(round(t*100))}"] = {"merged": m, "bg_frac": f, "target": t}
    names60 = lines(f"{U}/cls_context60.txt"); stuffn = lines(f"{U}/cls_coco_stuff.txt")
    assert len(names60) == 60 and len(stuffn) == 171
    for k, v in rep["ctx_variants"].items():
        v["merged_names"] = [names60[c] for c in v["merged"]]
    for k, v in rep["obj_variants"].items():
        v["merged_names"] = [stuffn[80 + (c - 91)] for c in v["merged"]]
    json.dump(rep, open(f"{S1}/tier1_bgfrac.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k in ("voc21", "context60", "coco_object")}))
    for k, v in {**rep["ctx_variants"], **rep["obj_variants"]}.items():
        print(k, round(v["bg_frac"], 4), len(v["merged"]), v["merged_names"][:12])


def stage_b():
    """Write derived label maps (compact ids) for frozen variants."""
    rep = json.load(open(f"{S1}/tier1_bgfrac.json"))
    cids = open(f"{CTX}/ImageSets/SegmentationContext/val.txt").read().split()
    for k, v in rep["ctx_variants"].items():
        root = f"{S1}/data/{k}"
        os.makedirs(f"{root}/SegmentationClassContext", exist_ok=True)
        os.makedirs(f"{root}/ImageSets/SegmentationContext", exist_ok=True)
        if not os.path.exists(f"{root}/JPEGImages"):
            os.symlink(os.path.realpath(f"{CTX}/JPEGImages"), f"{root}/JPEGImages")
        open(f"{root}/ImageSets/SegmentationContext/val.txt", "w").write(open(f"{CTX}/ImageSets/SegmentationContext/val.txt").read())
        keep = [c for c in range(1, 60) if c not in v["merged"]]
        lut = np.full(256, 255, np.uint8); lut[0] = 0
        for c in v["merged"]:
            lut[c] = 0
        for n, c in enumerate(keep):
            lut[c] = n + 1
        h = np.zeros(256, np.int64)
        for i in cids:
            a = lut[np.array(Image.open(f"{CTX}/SegmentationClassContext/{i}.png"))]
            h += np.bincount(a.ravel(), minlength=256)
            Image.fromarray(a).save(f"{root}/SegmentationClassContext/{i}.png")
        v["keep"] = keep; v["n_classes"] = len(keep) + 1; v["check_bg_frac"] = float(h[0] / h[:255].sum())
        v["check_pixels"] = int(h.sum()); v["check_max_label"] = int(np.nonzero(h)[0].max())
        print(k, v["n_classes"], v["check_bg_frac"], v["check_max_label"])
    fs = sorted(os.listdir(STUFF))
    for k, v in rep["obj_variants"].items():
        root = f"{S1}/data/{k}"
        os.makedirs(f"{root}/annotations/val2017", exist_ok=True)
        os.makedirs(f"{root}/images", exist_ok=True)
        if not os.path.exists(f"{root}/images/val2017"):
            os.symlink(os.path.realpath(f"{COCOO}/images/val2017"), f"{root}/images/val2017")
        keep = [c for c in STUFF_IDS if c not in v["merged"]]
        lut = np.zeros(256, np.uint8)  # stuff merged and 255 -> 0
        for tr, c in enumerate(THING):
            lut[c] = tr + 1
        for n, c in enumerate(keep):
            lut[c] = 81 + n
        h = np.zeros(256, np.int64)
        for f in fs:
            a = lut[np.array(Image.open(f"{STUFF}/{f}"))]
            h += np.bincount(a.ravel(), minlength=256)
            Image.fromarray(a).save(f"{root}/annotations/val2017/{f[:-4]}_instanceTrainIds.png")
        v["keep"] = keep; v["n_classes"] = 81 + len(keep); v["check_bg_frac"] = float(h[0] / h.sum())
        v["check_pixels"] = int(h.sum()); v["check_max_label"] = int(np.nonzero(h)[0].max())
        print(k, v["n_classes"], v["check_bg_frac"], v["check_max_label"])
    json.dump(rep, open(f"{S1}/tier1_bgfrac.json", "w"), indent=1)


def miou(I, Un):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(I.sum(0) / Un.sum(0)) * 100)


def stage_c():
    """H2: per-image bg fraction quintiles, O vs P* per bin."""
    from scipy.stats import spearmanr
    src = {"voc21": f"{A}/p2/results/full", "coco_object": f"{A}/p2/results/full", "context60": f"{A}/p5/cells"}
    out = {}
    for ds, d in src.items():
        lab = np.load(f"{d}/SCLIP__O__{ds}_perimage.npz")["label"]
        fr = lab[:, 0] / np.maximum(lab.sum(1), 1)
        order = np.argsort(fr, kind="stable")
        bins = np.array_split(order, 5)
        gaps = np.zeros((7, 5))
        for mi, m in enumerate(M7):
            o = np.load(f"{d}/{m}__O__{ds}_perimage.npz"); p = np.load(f"{d}/{m}__P__{ds}_perimage.npz")
            assert np.array_equal(o["label"], lab) and np.array_equal(p["label"], lab)
            for b, idx in enumerate(bins):
                gaps[mi, b] = miou(o["intersect"][idx], o["union"][idx]) - miou(p["intersect"][idx], p["union"][idx])
        mean = gaps.mean(0)
        rho = float(spearmanr(np.arange(5), mean).correlation)
        out[ds] = {"bin_bg_frac_mean": [float(fr[b].mean()) for b in bins], "bin_n": [len(b) for b in bins],
                   "gap_mean_per_bin": mean.tolist(), "gap_per_method": {m: gaps[i].tolist() for i, m in enumerate(M7)},
                   "spearman_bin_vs_gap": rho, "pass_0.9": rho >= 0.9}
        print(ds, [round(x, 3) for x in out[ds]["bin_bg_frac_mean"]], [round(x, 2) for x in mean], round(rho, 3))
    n = sum(v["pass_0.9"] for v in out.values())
    out["H2"] = "SUPPORTED" if n >= 2 else "NOT_SUPPORTED"
    out["n_datasets_pass"] = n
    json.dump(out, open(f"{S1}/tier1_h2.json", "w"), indent=1)
    print("H2", out["H2"], n)


if __name__ == "__main__":
    {"a": stage_a, "b": stage_b, "c": stage_c}[sys.argv[1]]()
