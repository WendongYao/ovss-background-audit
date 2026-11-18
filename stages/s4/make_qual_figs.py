"""S4: qualitative figures from the rendered SCLIP predictions (no GPU).
Labels under each prediction are the FROZEN per-image values from selection*.json (contract: numbers come from frozen npz).
Usage: python make_qual_figs.py <project_root> {main|supp}"""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT, which = sys.argv[1], sys.argv[2]
R = f"{ROOT}/experiments/S4/results"
OUT = f"{ROOT}/paper/figures"
pre = "" if which == "main" else "s"
sel = json.load(open(f"{R}/{'selection' if which == 'main' else 'selection_supp'}.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42})
INK, INK2 = "#0b0b0b", "#52514e"
# THR / NC predictions are taken from the RTX 3090 re-run (deviations 6-7, 9), which reproduces the frozen S3 statistics
# exactly; a dagger would mark any shown cell that still failed the gate (none do).
_r = json.load(open(f"{R}/s4b_check.json"))["R"]
FLAG = {(x["group"], x["setting"], x["dataset"], x["index"]) for x in _r if not x["pass3090"]}


def palette(n=256):
    p = np.zeros((n, 3), np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        p[i] = (r, g, b)
    p[255] = (255, 255, 255)  # ignore
    return p


PAL = palette()


def colour(path):
    return PAL[np.array(Image.open(path))]


def img(grp, ds, i):
    return np.array(Image.open(glob.glob(f"{R}/render/out/{grp}/img/{ds}__{i}__*")[0]).convert("RGB"))


def pred(grp, s, ds, i):
    if s in ("THR", "NC"):
        return colour(f"{R}/render_b/out/r3090/{grp}/SCLIP__{s}__{ds}_orig__{i}__pred.png")
    return colour(f"{R}/render/out/{grp}/SCLIP__{s}__{ds}__{i}__pred.png")


def gt(grp, ds, i):
    return colour(f"{R}/render/out/{grp}/SCLIP__P__{ds}__{i}__gt.png" if grp.endswith("q1")
                  else f"{R}/render/out/{grp}/SCLIP__BGB__{ds}__{i}__gt.png")


def grid(rows, cols, titles, fname, legend=None, width=7.1):
    ratios = [r["img"].shape[0] / r["img"].shape[1] for r in rows]
    cw = width / len(cols) * 0.97
    fig = plt.figure(figsize=(width, sum(ratios) * cw + 0.2 * len(rows) + 0.35 + (0.2 if legend else 0)))
    gs = fig.add_gridspec(len(rows), len(cols), height_ratios=ratios, hspace=0.18, wspace=0.03,
                          left=0.07, right=0.995, top=1 - 0.25 / fig.get_figheight(),
                          bottom=(0.32 if legend else 0.12) / fig.get_figheight())
    for a, r in enumerate(rows):
        for b, c in enumerate(cols):
            ax = fig.add_subplot(gs[a, b])
            ax.imshow(r[c], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#b8b7b2")
            if a == 0:
                ax.set_title(titles[b], fontsize=7.5, color=INK, pad=3)
            if c in r["lab"]:
                ax.set_xlabel(r["lab"][c], fontsize=6.5, color=INK2, labelpad=1.5)
            if b == 0:
                ax.set_ylabel(r["name"], fontsize=6.5, color=INK, labelpad=3)
    if legend:
        fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=6.5, frameon=False,
                   handlelength=1.0, borderaxespad=0.1)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", fname)


DSN = {"voc21": "VOC21", "coco_object": "COCO-Obj."}
SET = ["P", "THR", "BGB", "O"]
rows = []
for ds in ("voc21", "coco_object"):
    for s in sel["Q1"][ds]["selected"]:
        i = s["index"]
        r = {"img": img(pre + "q1", ds, i), "gt": gt(pre + "q1", ds, i),
             "name": f"{DSN[ds]} p{s['percentile']}",
             "lab": {k: f"{s['miou'][k]:.1f}" + ("\u2020" if (pre + "q1", k, ds, i) in FLAG else "") for k in SET}}
        for k in SET:
            r[k] = pred(pre + "q1", k, ds, i)
        rows.append(r)
cols = ["img", "gt"] + SET
titles = ["Image", "GT", "P*", "THR", "BGB", "O"]
if which == "main":
    grid(rows, cols, titles, "fig6_qualitative_settings")
else:
    for ds, part in (("voc21", rows[:10]), ("coco_object", rows[10:])):
        grid(part, cols, titles, f"figS_qualitative_settings_{ds}")

rows = []
for s in sel["Q2"]["coco_object"]["selected"]:
    i = s["index"]
    rows.append({"img": img(pre + "q2", "coco_object", i), "gt": gt(pre + "q2", "coco_object", i),
                 "BGB": pred(pre + "q2", "BGB", "coco_object", i), "NC": pred(pre + "q2", "NC", "coco_object", i),
                 "name": f"p{s['percentile']}",
                 "lab": {k: f"bed IoU {s['bed_iou'][k]:.1f}" for k in ("BGB", "NC")}})
bed = [Patch(facecolor=PAL[60] / 255, edgecolor="none", label="bed"),
       Patch(facecolor=PAL[0] / 255, edgecolor="none", label="background")]
if which == "main":
    grid(rows, ["img", "gt", "BGB", "NC"], ["Image", "GT", "BGB", "NC"], "fig7_qualitative_collision", legend=bed, width=4.6)
else:  # 10 rows: two halves of 5 rows, placed side by side in LaTeX
    grid(rows[:5], ["img", "gt", "BGB", "NC"], ["Image", "GT", "BGB", "NC"], "figS_qualitative_collision_a", legend=bed, width=4.6)
    grid(rows[5:], ["img", "gt", "BGB", "NC"], ["Image", "GT", "BGB", "NC"], "figS_qualitative_collision_b", legend=bed, width=4.6)
