"""V5 scoring per CONTRACT_CN.md. Q1 on Context-60 (O vs P*), Q2 Delta_trick(CF) over 21 cells, descriptive reads."""
import itertools
import json
import os

W = "/root/autodl-tmp/p5"
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
# phase-2 BGB and P* (p2_table_transcribed.csv)
P2 = {"BGB": {"voc21": dict(zip(M7, [60.76, 51.80, 57.02, 60.35, 64.34, 61.27, 67.08])),
              "coco_object": dict(zip(M7, [33.07, 33.04, 32.34, 34.56, 36.75, 36.69, 41.29]))},
      "P": {"voc21": dict(zip(M7, [35.91, 37.77, 36.79, 36.66, 42.07, 40.57, 40.02])),
            "coco_object": dict(zip(M7, [21.68, 23.32, 21.55, 22.00, 24.74, 23.89, 24.90]))}}


def cell(m, tag, ds):
    p = f"{W}/cells/{m}__{tag}__{ds}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def p4(m, ds, tag):
    p = f"/root/autodl-tmp/{'p3' if tag == 'SEL' else 'p4'}/test/{m}__{ds}__{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def r2(x):
    return None if x is None else round(x, 3)


res = {"missing": []}
# ---------- Q1
O = {m: cell(m, "O", "context60")["mIoU"] for m in M7}
P = {m: cell(m, "P", "context60")["mIoU"] for m in M7}
d = {m: round(O[m] - P[m], 2) for m in M7}
mean_d = sum(d.values()) / 7
disc = [(a, b) for a, b in itertools.combinations(M7, 2) if (O[a] - O[b]) * (P[a] - P[b]) < 0]
a_ok, b_ok = mean_d >= 5.0, len(disc) >= 3
res["Q1"] = {"O": O, "P": P, "O_minus_P": d, "mean": r2(mean_d), "discordant": len(disc), "pairs": disc,
             "a": a_ok, "b": b_ok, "decision": "GO" if a_ok and b_ok else ("GO_NARROW" if a_ok else "STOP")}
g0 = {"SCLIP": 31.74, "ClearCLIP": 32.6, "NACLIP": 34.99, "ResCLIP": 33.5, "SC-CLIP": 36.8, "ProxyCLIP": 35.3, "Trident": 38.6}
res["G0"] = {m: [O[m], g0[m], round(O[m] - g0[m], 2)] for m in M7}
# ---------- Q2 and descriptive
rows = []
for ds in ["voc21", "coco_object", "context60"]:
    for m in M7:
        cf = cell(m, "CF", ds)
        if cf is None:
            res["missing"].append(f"{m} CF {ds}")
            continue
        if ds == "context60":
            bgb, pp = cell(m, "BGB", ds)["mIoU"], P[m]
            sel = cell(m, "SEL26", ds)["mIoU"]
            cc = None
        else:
            bgb, pp = P2["BGB"][ds][m], P2["P"][ds][m]
            s, c = p4(m, ds, "SEL"), p4(m, ds, "CC")
            sel = s["mIoU"] if s else None
            cc = c["mIoU"] if c else None
        rows.append({"method": m, "dataset": ds, "CF": cf["mIoU"], "BGB": bgb, "P": pp, "SEL26": sel, "CC": cc,
                     "n_bg_words": cf.get("n_bg_words"), "d_trick": round(cf["mIoU"] - bgb, 2),
                     "d_plain": round(cf["mIoU"] - pp, 2), "d_vs26": None if sel is None else round(cf["mIoU"] - sel, 2),
                     "d_vsCC": None if cc is None else round(cf["mIoU"] - cc, 2)})
mean = lambda xs: sum(xs) / len(xs) if xs else None
dt = mean([r["d_trick"] for r in rows])
per = {ds: {k: r2(mean([r[k] for r in rows if r["dataset"] == ds and r[k] is not None]))
            for k in ["d_trick", "d_plain", "d_vs26", "d_vsCC"]} for ds in ["voc21", "coco_object", "context60"]}
voc_dt = per["voc21"]["d_trick"]
adopt = len(rows) == 21 and dt >= -2.0 and voc_dt is not None and voc_dt >= -2.0
res["Q2"] = {"n_cells": len(rows), "delta_trick_CF": r2(dt), "voc21_delta_trick_CF": voc_dt,
             "decision": "ADOPT_CF_BASELINE" if adopt else "REPORT_VOCAB_SENSITIVITY", "per_dataset": per,
             "n_ge_trick": sum(r["d_trick"] >= 0 for r in rows)}
# Context-60 SEL26 vs P (descriptive)
res["context60_SEL26_minus_P"] = {m: round(cell(m, "SEL26", "context60")["mIoU"] - P[m], 2) for m in M7}
res["context60_BGB_minus_O"] = {m: round(cell(m, "BGB", "context60")["mIoU"] - O[m], 2) for m in M7}
# VOC21 per-class: CF vs SEL26 and CF vs CC, mean over methods
names = [l.split(",")[0].strip() for l in open("/root/autodl-tmp/p2/repos/ClearCLIP/configs/cls_voc21.txt").read().split("\n") if l.strip()]
pc = {}
for tag in ["SEL", "CC"]:
    diffs = []
    for m in M7:
        cf, ref = cell(m, "CF", "voc21"), p4(m, "voc21", tag)
        if cf and ref:
            fa = 100.0 if max(cf["iou"]) <= 1.0 else 1.0
            fb = 100.0 if max(ref["iou"]) <= 1.0 else 1.0
            diffs.append([a * fa - b * fb for a, b in zip(cf["iou"], ref["iou"])])
    if diffs:
        avg = [sum(col) / len(col) for col in zip(*diffs)]
        pc[tag] = sorted([[n, round(v, 1)] for n, v in zip(names, avg)], key=lambda x: x[1])
res["voc21_perclass_CF_minus"] = pc
j = json.load(open(f"{W}/cf_judgements.json", encoding="utf-8"))
res["removed_words"] = {ds: [w for w, a in j[ds].items() if a == "non-ascii" or a.startswith("yes")] for ds in j}
res["rows"] = rows
json.dump(res, open(f"{W}/p5_score.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: res[k] for k in ["Q1", "Q2", "missing", "context60_SEL26_minus_P", "context60_BGB_minus_O"]}, ensure_ascii=False))
print(json.dumps(pc, ensure_ascii=False))
