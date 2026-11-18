import csv, json, sys
import numpy as np
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
s3 = {(r["method"], r["variant"].replace("_orig", ""), r["setting"]): float(r["mIoU"]) for r in csv.DictReader(open(sys.argv[1]))}
p2 = {(r["method"], r["dataset"], r["setting"]): float(r["mIoU"]) for r in csv.DictReader(open(sys.argv[2]))}
zero = {(m, "voc21") for m in M7} | {("ClearCLIP", "coco_object")}
out = {}
for d in ["voc21", "coco_object"]:
    rows = {}
    for m in M7:
        P, B, O, T = p2[(m, d, "P")], p2[(m, d, "BGB")], p2[(m, d, "O")], s3[(m, d, "THR")]
        N = B if (m, d) in zero else s3[(m, d, "NC")]
        rows[m] = {"P": P, "THR": T, "BGB": B, "NC": N, "O": O, "d_BG": round(B - P, 2), "d_thr": round(T - P, 2),
                   "d_name": round(B - T, 2), "d_voc": round(N - B, 2), "zero": (m, d) in zero}
    mb = np.mean([rows[m]["d_BG"] for m in M7]); mt = np.mean([rows[m]["d_thr"] for m in M7])
    share = mt / mb
    reading = "mainly_threshold" if share >= 0.67 else ("mainly_names" if share <= 0.33 else "both")
    mv = np.mean([rows[m]["d_voc"] for m in M7]); npos = sum(rows[m]["d_voc"] > 0 for m in M7 if not rows[m]["zero"])
    out[d] = {"rows": rows, "mean_d_BG": round(mb, 3), "mean_d_thr": round(mt, 3), "mean_d_name": round(float(np.mean([rows[m]["d_name"] for m in M7])), 3),
              "share_thr": round(share, 3), "reading": reading, "mean_d_voc": round(mv, 3), "n_pos_voc_among_colliding": int(npos),
              "collision_reading": ("measurable" if (mv >= 0.5 and npos >= 4) else "small_or_none") if d == "coco_object" else "descriptive"}
json.dump(out, open(sys.argv[3], "w"), indent=1)
for d, v in out.items():
    print(d, {k: v[k] for k in v if k != "rows"})
    for m, r in v["rows"].items(): print("  ", m, r["P"], r["THR"], r["BGB"], r["NC"], "dthr", r["d_thr"], "dname", r["d_name"], "dvoc", r["d_voc"])
