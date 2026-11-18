"""S4 deviation 6: thumbnails for the overview figure (fig0, option B) from the 3090 renders of bed images 369 / 4575.
Writes paper/figures/fig0_thumbs/<idx>/*.png and a contact sheet fig0_thumbs/contact_<idx>.png (preview only).
Usage: python make_fig0_thumbs.py <project_root>"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = sys.argv[1]
B = f"{ROOT}/experiments/S4/results/render_b/out/b"
OUT = f"{ROOT}/paper/figures/fig0_thumbs"
V = {"P": "coco_object", "THR": "coco_object_orig", "BGB": "coco_object", "NC": "coco_object_orig", "O": "coco_object"}
BED = 60


def palette(n=256):
    p = np.zeros((n, 3), np.uint8)
    for i in range(n):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j); g |= ((c >> 1) & 1) << (7 - j); b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        p[i] = (r, g, b)
    p[255] = (255, 255, 255)
    return p


PAL = palette()
CMAP = plt.get_cmap("magma")


def save(a, path, w=360):
    im = Image.fromarray(a)
    h = round(im.height * w / im.width)
    im.resize((w, h), Image.NEAREST if a.ndim == 3 and a.dtype == np.uint8 and path.endswith("_lab.png") else Image.BILINEAR).save(path)


for i in (369, 4575):
    o = f"{OUT}/{i}"
    os.makedirs(o, exist_ok=True)
    img = np.array(Image.open(glob.glob(f"{B}/img/coco_object__{i}__*")[0]).convert("RGB"))
    save(img, f"{o}/input.png")
    gt = np.array(Image.open(f"{B}/SCLIP__P__coco_object__{i}__gt.png"))
    save(PAL[gt], f"{o}/gt_lab.png")
    tiles = [("input", img), ("GT", PAL[gt])]
    for s, v in V.items():
        pred = np.array(Image.open(f"{B}/SCLIP__{s}__{v}__{i}__pred.png"))
        save(PAL[pred], f"{o}/pred_{s}_lab.png")
        d = np.load(f"{B}/SCLIP__{s}__{v}__{i}__inter.npz")
        cls = list(d["P_classes"])
        Pbg = d["P"][cls.index(0)].astype(np.float32)
        Pbed = d["P"][cls.index(BED)].astype(np.float32)
        for name, m in (("Pbg", Pbg), ("Pbed", Pbed), ("conf", d["conf"].astype(np.float32))):
            save((CMAP(np.clip(m, 0, 1))[..., :3] * 255).astype(np.uint8), f"{o}/{name}_{s}.png")
        save(PAL[d["argmax_raw"]], f"{o}/argmax_raw_{s}_lab.png")
        tiles += [(f"{s} P(bg)", (CMAP(np.clip(Pbg, 0, 1))[..., :3] * 255).astype(np.uint8)),
                  (f"{s} P(bed)", (CMAP(np.clip(Pbed, 0, 1))[..., :3] * 255).astype(np.uint8)),
                  (f"{s} argmax", PAL[d["argmax_raw"]]), (f"{s} final", PAL[pred])]
    n = len(tiles); cols = 4 + 0
    rows = (n - 2 + 3) // 4 + 1
    fig, axes = plt.subplots(rows, 4, figsize=(8, 2.1 * rows))
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].imshow(tiles[0][1]); axes[0, 0].set_title(tiles[0][0], fontsize=7)
    axes[0, 1].imshow(tiles[1][1]); axes[0, 1].set_title(tiles[1][0], fontsize=7)
    for k, (t, a) in enumerate(tiles[2:]):
        ax = axes[1 + k // 4, k % 4]
        ax.imshow(a); ax.set_title(t, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/contact_{i}.png", dpi=110)
    plt.close(fig)
    print("thumbs", i)
