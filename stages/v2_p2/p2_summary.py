"""Compact summary of full_score.json + a CSV of all cells. Usage: python p2_summary.py <score_json> <results_dir> <out_csv>"""
import glob
import json
import sys

s = json.load(open(sys.argv[1]))
rows = []
for f in sorted(glob.glob(f"{sys.argv[2]}/*__*__*.json")):
    r = json.load(open(f))
    rows.append([r["method"], r["setting"], r["dataset"], f"{r['mIoU']:.2f}", f"{r['mIoU_recomputed']:.2f}",
                 r["n_images"], f"{r['seconds']:.0f}", r["short_side"]])
with open(sys.argv[3], "w") as fo:
    fo.write("method,setting,dataset,mIoU,mIoU_recomputed,n_images,seconds,resize_scale\n")
    for x in rows:
        fo.write(",".join(str(v).replace(",", ";") for v in x) + "\n")
print("STATUS", s["status"], "cells", s["cells_found"], "new_fail", s["new_methods_failing_gate"])
for m, g in s["gate"].items():
    print("G0", m, g["pass"], " ".join(f"{d}:{v:+.2f}" for d, v in g["diffs"].items()))
t = s["T1"]
print("T1 holds", t["holds"], "G_rep %.3f G_fix %.3f p90 %.3f" % (t["G_rep"], t["G_fix"], t["G_fix_p90"]), t["later_methods"])
print("T1 per-dataset", {d: round(v, 2) for d, v in t["per_dataset_fix"].items()})
if "T1_vfm_descriptive" in s:
    print("T1 vfm", {k: round(v, 3) for k, v in s["T1_vfm_descriptive"].items()})
t2 = s["T2"]
print("T2 holds", t2["holds"], "otsu-trick %.3f otsu-plain %.3f" % (t2["otsu_minus_official_trick"], t2["otsu_minus_plain"]))
for r in t2["rows"]:
    print("T2row", r["method"], r["dataset"], "P %.2f BGB %.2f OTSU %.2f" % (r["P"], r["BGB"], r["OTSU"]))
print("mismatch_recomputed", [x[:3] for x in rows if abs(float(x[3]) - float(x[4])) > 0.01])
