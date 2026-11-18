"""Make all paper figures from the project's verified result files (no hand-typed numbers except where noted).
Run from the project root:  python paper/scripts/make_figures.py  (needs numpy, matplotlib; archives extracted to DATA).
Usage: python make_figures.py PROJECT_ROOT DATA_DIR OUT_DIR
  PROJECT_ROOT: folder OVSS_Journal_20261107 (inputs/, experiments/)
  DATA_DIR: folder where archives/p3_results, p4_dir, p5_results tarballs are extracted (for per-class IoU)
  OUT_DIR: paper/figures
Palette: categorical slots 1-3 of the reference palette (validated all-pairs, light mode); neutral grays for context."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT, DATA, OUT = sys.argv[1:4]
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29d", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})
W = 6.3  # inches, single-column text width (cas-sc)


def p2():
    t = {}
    for r in csv.DictReader(open(f"{ROOT}/inputs/Probe_OVSSProtocol02/full_score/p2_table_transcribed.csv")):
        t[(r["method"], r["setting"], r["dataset"])] = float(r["mIoU"])
    for r in csv.DictReader(open(f"{ROOT}/inputs/Probe_OVSSContext05/results/p5_cells_transcribed.csv")):
        if r["dataset"] == "context60":
            for s in ("P", "BGB", "SEL26", "CF"):
                t[(r["method"], s, "context60")] = float(r[s])
    for m in M7:  # Context-60 official (O) cells from the V5 result package
        t[(m, "O", "context60")] = round(json.load(open(f"{DATA}/p5_results/p5/cells/{m}__O__context60.json"))["mIoU"], 2)
    return t


T = p2()
CI = json.load(open(f"{ROOT}/experiments/analysis/ci_bootstrap.json"))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Fig. 1: protocol gap O - P* per dataset (methods as gray dots, mean +- 95% CI)
DS = [("voc21", "VOC21"), ("coco_object", "COCO-Object"), ("voc20", "VOC20"), ("context60", "Context-60"), ("ade20k", "ADE20K")]
fig, ax = plt.subplots(figsize=(W, 2.4))
for i, (d, lab) in enumerate(DS):
    g = [T[(m, "O", d)] - T[(m, "P", d)] for m in M7]
    ax.scatter(np.full(7, i) + np.linspace(-0.18, 0.18, 7), g, s=14, color=MUTED, zorder=2, linewidths=0)
    c = CI[f"O-P | {d}"]
    ax.errorbar(i + 0.32, c["mean_point"], yerr=[[c["mean_point"] - c["mean_ci95"][0]], [c["mean_ci95"][1] - c["mean_point"]]],
                fmt="o", ms=5, color=C1, ecolor=C1, elinewidth=1.2, capsize=2, zorder=3)
    ax.annotate(f"{c['mean_point']:.1f}", (i + 0.42, c["mean_point"]), va="center", fontsize=7, color=INK)
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xticks(range(len(DS)), [l for _, l in DS]); ax.set_ylabel("O − P* (mIoU)"); ax.grid(axis="x", visible=False)
ax.set_xlim(-0.5, len(DS) - 0.2)
ax.text(0.99, 0.95, "gray: 7 methods   blue: mean, 95% CI", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
save(fig, "fig1_protocol_gap")

# Fig. 2: background share vs mean BGB - P* (S1); three series
S1 = json.load(open(f"{ROOT}/experiments/S1/results/s1_h1_score.json"))["points"]
fig, ax = plt.subplots(figsize=(W * 0.62, 2.8))
groups = [("Original datasets", ["context60", "coco_object", "voc21"], C1, "o"),
          ("Context-60 variants", ["context60", "ctx_bg30", "ctx_bg50", "ctx_bg70"], C2, "s"),
          ("COCO-Stuff variants", ["obj_bg30", "obj_bg50", "coco_object"], C3, "^")]
lab = {"voc21": "VOC21", "coco_object": "COCO-Object", "context60": "Context-60", "ctx_bg30": "35%", "ctx_bg50": "53%",
       "ctx_bg70": "71%", "obj_bg30": "31%", "obj_bg50": "51%"}  # actual shares (Appendix B), match Table A2
for name, pts, col, mk in groups:
    x = [S1[p]["bg_frac"] for p in pts]; y = [S1[p]["mean_dBG"] for p in pts]
    if name != "Original datasets":
        ax.plot(x, y, color=col, lw=1.5, zorder=2)
    for p, xx, yy in zip(pts, x, y):
        k = f"BGB-P | {p}"; c = CI[k]
        ax.errorbar(xx, yy, yerr=[[yy - c["mean_ci95"][0]], [c["mean_ci95"][1] - yy]], fmt=mk, ms=6 if name == "Original datasets" else 5,
                    color=col, ecolor=col, elinewidth=1, capsize=0, zorder=3, mec="white", mew=0.8)
    ax.plot([], [], marker=mk, color=col, ls="-" if name != "Original datasets" else "none", label=name)
for p, dx, dy in [("voc21", -0.02, 1.2), ("coco_object", -0.1, -0.6), ("context60", 0.015, 1.1), ("ctx_bg70", 0.01, 1.0),
                  ("obj_bg30", 0.0, 1.0), ("obj_bg50", 0.035, -1.6)]:
    ax.annotate(lab[p] if p in ("voc21", "coco_object", "context60") else ("Ctx " + lab[p] if p.startswith("ctx") else "COCO " + lab[p]),
                (S1[p]["bg_frac"] + dx, S1[p]["mean_dBG"] + dy), fontsize=7, color=INK, ha="center")
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xlabel("Background pixel share"); ax.set_ylabel("BGB − P* (mIoU, mean of 7)")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(loc="upper left", fontsize=7)
save(fig, "fig2_bgshare_scatter")

# Fig. 3: rank change O -> P* (bump chart), three datasets
fig, axs = plt.subplots(1, 3, figsize=(W, 2.3), sharey=True)
for ax, (d, lab3) in zip(axs, [("voc21", "VOC21"), ("coco_object", "COCO-Object"), ("context60", "Context-60")]):
    rO = {m: r for r, m in enumerate(sorted(M7, key=lambda m: -T[(m, "O", d)]), 1)}
    rP = {m: r for r, m in enumerate(sorted(M7, key=lambda m: -T[(m, "P", d)]), 1)}
    for m in M7:
        ax.plot([0, 1], [rO[m], rP[m]], color=C1 if rO[m] != rP[m] else MUTED, lw=1.2, marker="o", ms=3.5)
        ax.text(-0.06, rO[m], m, ha="right", va="center", fontsize=6.5, color=INK)
        ax.text(1.06, rP[m], m, ha="left", va="center", fontsize=6.5, color=INK)
    disc = sum((T[(a, "O", d)] - T[(b, "O", d)]) * (T[(a, "P", d)] - T[(b, "P", d)]) < 0 for i, a in enumerate(M7) for b in M7[i + 1:])
    ax.set_title(f"{lab3}: {disc}/21 swaps", fontsize=8, color=INK)
    ax.set_xticks([0, 1], ["O", "P*"]); ax.set_xlim(-0.9, 1.9)
    ax.grid(False); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", left=False, labelleft=False)
axs[0].invert_yaxis(); fig.subplots_adjust(wspace=0.35)
save(fig, "fig3_rank_change")

# Fig. 4: per-class IoU change on VOC21 (7-method mean), LLM vocabulary vs 26-word list
names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
         "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]  # mmseg VOC order
ld = lambda f: np.array(json.load(open(f))["iou"], float)
sel = np.array([ld(f"{DATA}/p3_results/p3/test/{m}__voc21__SEL.json") for m in M7])
cc = np.array([ld(f"{DATA}/p4_dir/p4/test/{m}__voc21__CC.json") for m in M7])
cf = np.array([ld(f"{DATA}/p5_results/p5/cells/{m}__CF__voc21.json") for m in M7])
s = 100 if np.nanmax(sel) <= 1 else 1
d_cc, d_cf = (cc - sel).mean(0) * s, (cf - sel).mean(0) * s
order = np.argsort(d_cc)
fig, ax = plt.subplots(figsize=(W * 0.62, 3.6))
y = np.arange(len(names))
ax.barh(y - 0.2, d_cc[order], height=0.38, color=C1, label="LLM vocabulary (CC) − 26 words")
ax.barh(y + 0.2, d_cf[order], height=0.38, color=C2, label="Conflict-filtered (CF) − 26 words")
ax.set_yticks(y, [names[i] for i in order], fontsize=7); ax.axvline(0, color=MUTED, lw=0.6)
ax.set_xlabel("Δ IoU on VOC21 (mean of 7 methods)"); ax.grid(axis="y", visible=False); ax.legend(loc="upper left", fontsize=7)
save(fig, "fig4_perclass_collision")

# Fig. 5: decomposition of the official background trick on original datasets (S3)
S3 = json.load(open(f"{ROOT}/experiments/S3/results/s3_score.json"))
fig, axs = plt.subplots(1, 2, figsize=(W, 2.4), sharey=False)
for ax, (d, lab5) in zip(axs, [("voc21", "VOC21"), ("coco_object", "COCO-Object")]):
    R = S3[d]["rows"]; x = np.arange(7)
    thr = [R[m]["d_thr"] for m in M7]; nam = [R[m]["d_name"] for m in M7]
    ax.bar(x, thr, width=0.62, color=C1, label="numeric threshold part (THR − P*)")
    ax.bar(x, nam, width=0.62, bottom=thr, color=C2, label="official name file (BGB − THR)", edgecolor="white", linewidth=1)
    ax.set_xticks(x, M7, rotation=35, ha="right", fontsize=7); ax.grid(axis="x", visible=False)
    ax.set_title(f"{lab5}: threshold share {S3[d]['share_thr']:.2f}", fontsize=8, color=INK)
axs[0].set_ylabel("BGB − P* (mIoU)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.08))
save(fig, "fig5_trick_decomposition")
print("figures written to", OUT)
