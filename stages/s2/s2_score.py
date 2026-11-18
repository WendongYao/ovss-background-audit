"""S2 scoring per experiments/CONTRACT_S2_CN.md. Inputs: S2 cells (THR/NC) and S1 cells (P/BGB)."""
import csv, json, sys
import numpy as np
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
s2 = {(r["method"], r["variant"], r["setting"]): float(r["mIoU"]) for r in csv.DictReader(open(sys.argv[1]))}
s1 = {(r["method"], r["variant"], r["setting"]): float(r["mIoU"]) for r in csv.DictReader(open(sys.argv[2]))}
zero = {("ClearCLIP", "obj_bg30"), ("ClearCLIP", "obj_bg50")} | {(m, "ctx_bg70") for m in M7}  # s2_collisions.json
out = {"layers": {}}
for v in ["obj_bg30", "obj_bg50", "ctx_bg70"]:
    L = {}
    for m in M7:
        P, BGB = s1[(m, v, "P")], s1[(m, v, "BGB")]
        thr = s2.get((m, v, "THR"))
        nc = BGB if (m, v) in zero else s2.get((m, v, "NC"))
        L[m] = {"P": P, "BGB": BGB, "THR": thr, "NC": nc, "zero_collision": (m, v) in zero,
                "d_voc": None if nc is None else round(nc - BGB, 2), "d_thr": None if thr is None else round(thr - P, 2),
                "NC_minus_P": None if nc is None else round(nc - P, 2)}
    out["layers"][v] = L
A = [out["layers"][v][m]["d_voc"] for v in ["obj_bg30", "obj_bg50"] for m in M7]
complete_AB = all(x is not None for x in A)
if not all(out["layers"]["obj_bg30"][m]["d_voc"] is not None for m in M7):
    out["decision"] = "INSUFFICIENT"
else:
    a = float(np.mean(A)) >= 1.0
    b = sum(out["layers"]["obj_bg30"][m]["d_voc"] > 0 for m in M7) >= 5
    out.update({"mean_d_voc_AB": float(np.mean(A)), "n_pos_obj_bg30": int(sum(out["layers"]["obj_bg30"][m]["d_voc"] > 0 for m in M7)),
                "a": bool(a), "b": bool(b)})
    out["decision"] = "COLLISION_SUPPORTED" if a and b else ("PARTIAL" if a or b else "NOT_SUPPORTED")
for v in ["obj_bg30", "obj_bg50", "ctx_bg70"]:
    for k in ["d_voc", "d_thr", "NC_minus_P"]:
        xs = [out["layers"][v][m][k] for m in M7 if out["layers"][v][m][k] is not None]
        out.setdefault("means", {})[f"{v}:{k}"] = [round(float(np.mean(xs)), 3), len(xs)] if xs else None
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "layers"}, indent=1))
for v in out["layers"]:
    print(v, {m: (d["d_voc"], d["d_thr"], d["NC_minus_P"]) for m, d in out["layers"][v].items()})
