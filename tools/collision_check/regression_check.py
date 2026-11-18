"""Re-derive the paper's collision lists (S2: COCO-Stuff/Context variants; S3: original VOC21 and COCO-Object) with
ovss_collision_check and compare them with the released NC name files and *_collisions.json, byte for byte.

Usage: python regression_check.py --repos DIR_WITH_OFFICIAL_REPOS --run S2_OR_S3_DIR [--run ...]
Each run dir must contain names/<variant>__NC__<method>.txt and s2_collisions.json or s3_collisions.json.
The BGB input file is rebuilt as: official background line (first line of the method's official name file) +
the query lines of the NC file (NC changes only the first line)."""
import argparse
import glob
import json
import os

import ovss_collision_check as cc

OFFICIAL = {"obj_bg30": "cls_coco_object.txt", "obj_bg50": "cls_coco_object.txt", "ctx_bg70": "cls_context60.txt",
            "voc21_orig": "cls_voc21.txt", "coco_object_orig": "cls_coco_object.txt"}
SEP = {"ProxyCLIP": ";", "Trident": ";"}

ap = argparse.ArgumentParser()
ap.add_argument("--repos", required=True)
ap.add_argument("--run", action="append", required=True)
a = ap.parse_args()
n_ok = n_bad = 0
for run in a.run:
    js = (glob.glob(os.path.join(run, "s2_collisions.json")) if "s2" in os.path.basename(run.rstrip("/")) else []) \
        or glob.glob(os.path.join(run, "s3_collisions.json")) or glob.glob(os.path.join(run, "s2_collisions.json"))
    ref = json.load(open(js[0]))
    for variant, methods in ref.items():
        for m, r in methods.items():
            nc_path = os.path.join(run, "names", f"{variant}__NC__{m}.txt")
            if not os.path.exists(nc_path):
                print(f"skip {variant} {m}: no NC file (zero collisions, BGB reused)"); continue
            nc = cc.read_lines(nc_path)
            bg_line = cc.read_lines(os.path.join(a.repos, m, "configs", OFFICIAL[variant]))[0]
            rep = cc.check(bg_line, nc[1:], SEP.get(m, ","))
            same_text = (open(nc_path, encoding="utf-8").read() == "\n".join([rep["clean_background_line"]] + nc[1:]) + "\n")
            same_list = (rep["removed"] == r["removed"] and rep["n_background_words"] == r["n_bg_words"])
            ok = same_text and same_list
            n_ok += ok; n_bad += (not ok)
            print(f"{'OK ' if ok else 'BAD'} {variant:17s} {m:10s} bg={rep['n_background_words']:2d} removed={rep['n_removed']:2d}")
print(f"\n{n_ok} identical, {n_bad} different")
