"""Generate LaTeX tables (booktabs) from the project's verified result files.
Usage: python make_tables.py PROJECT_ROOT DATA_DIR OUT_DIR
DATA_DIR holds extracted archives: p2_results_core/, p3_results/, p4_dir/, p5_results/, s2r/, s3r/."""
import csv
import json
import sys

import numpy as np

ROOT, DATA, OUT = sys.argv[1:4]
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
FAM = {"SCLIP": "C", "ClearCLIP": "C", "NACLIP": "C", "ResCLIP": "C", "SC-CLIP": "C", "ProxyCLIP": "V", "Trident": "V"}

T = {}
for r in csv.DictReader(open(f"{ROOT}/inputs/Probe_OVSSProtocol02/full_score/p2_table_transcribed.csv")):
    T[(r["method"], r["setting"], r["dataset"])] = float(r["mIoU"])
for m in M7:
    for s in ("O", "P", "BGB", "SEL26", "CF"):
        T[(m, s, "context60")] = round(json.load(open(f"{DATA}/p5_results/p5/cells/{m}__{s}__context60.json"))["mIoU"], 2)
    for d in ("voc21", "coco_object"):
        T[(m, "SEL26", d)] = round(json.load(open(f"{DATA}/p3_results/p3/test/{m}__{d}__SEL.json"))["mIoU"], 2)
        T[(m, "CC", d)] = round(json.load(open(f"{DATA}/p4_dir/p4/test/{m}__{d}__CC.json"))["mIoU"], 2)
        T[(m, "CF", d)] = round(json.load(open(f"{DATA}/p5_results/p5/cells/{m}__CF__{d}.json"))["mIoU"], 2)
f2 = lambda x: f"{x:.2f}"
sg = lambda x: f"{x:+.2f}".replace("+", "$+$").replace("-", "$-$")


def write(name, s):
    open(f"{OUT}/{name}.tex", "w").write(s)


# Table 1: audit of official per-dataset configuration (background-related keys)
audit = json.load(open(f"{DATA}/p2_results_core/results/cfg_audit.json"))
col = {**json.load(open(f"{DATA}/s3r/s3/s3_collisions.json")), **json.load(open(f"{DATA}/s2r/s2/s2_collisions.json"))}
# Context-60 numeric keys read from each repository's configs/cfg_context60.py on 2026-11-07 (logit_scale only set by SCLIP)
CTX = {"SCLIP": ("0.10", "50"), "ClearCLIP": ("0.15", "--"), "NACLIP": ("0.10", "--"), "ResCLIP": ("0.10", "--"),
       "SC-CLIP": ("0.15", "--"), "ProxyCLIP": ("0.15", "--"), "Trident": ("0.15", "--")}
rows = []
for m in M7:
    v = audit[m]["varies"]
    nvar = len([k for k in v if k != "name_path"])
    g = lambda k, i: (v[k][i] if k in v and v[k][i] != "<absent>" else "--")
    rows.append(f"{m} & {nvar} & {g('prob_thd',0)} & {g('logit_scale',0)} & {g('area_thd',0)} & {col['voc21_orig'][m]['n_bg_words']} & "
                f"{g('prob_thd',2)} & {g('logit_scale',2)} & {col['coco_object_orig'][m]['n_bg_words']} & {CTX[m][0]} & {CTX[m][1]} & 1 \\\\")
write("tab1_config_audit", r"""\begin{table}[t]
\caption{Official per-dataset settings that touch background handling. ``\#var'' counts model keys (other than the class-name file) whose value differs across the official VOC21/VOC20/COCO-Object/ADE20K configs. $\tau$: probability threshold for background; $s$: logit scale; $a$: area threshold; ``\#bg'': number of background words in the official class-name file. ``--'': not set (class default).}\label{tab:audit}
\centering\small
\begin{tabular}{@{}lc|cccc|ccc|ccc@{}}
\toprule
 & & \multicolumn{4}{c|}{VOC21} & \multicolumn{3}{c|}{COCO-Object} & \multicolumn{3}{c}{Context-60} \\
Method & \#var & $\tau$ & $s$ & $a$ & \#bg & $\tau$ & $s$ & \#bg & $\tau$ & $s$ & \#bg \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Table 2: reproduction of reported numbers under official configs (O)
REP = {"SCLIP": [59.63, 81.54, 33.52, 16.45, 31.74], "ClearCLIP": [51.8, 80.9, 33.0, 16.7, 32.6], "NACLIP": [64.07, 83.03, 36.19, 19.05, 34.99],
       "ResCLIP": [61.1, 86.0, 35.0, 18.0, 33.5], "SC-CLIP": [64.6, 84.3, 37.7, 20.1, 36.8], "ProxyCLIP": [61.3, 80.3, 37.5, 20.2, 35.3],
       "Trident": [67.1, 84.5, 41.1, 21.9, 38.6]}  # targets fixed in the V2-P2 / V5 contracts before any run
DS5 = ["voc21", "voc20", "coco_object", "ade20k", "context60"]
rows = []
for m in M7:
    cells = []
    for i, d in enumerate(DS5):
        o = T[(m, "O", d)]; cells.append(f"{f2(o)} ({sg(o - REP[m][i])})")
    rows.append(f"{m} & " + " & ".join(cells) + r" \\")
write("tab2_reproduction", r"""\begin{table}[t]
\caption{Official configurations (O) re-run from the official code, with the difference to the number reported by each method in parentheses. Reported targets were fixed before running.}\label{tab:repro}
\centering\small
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & VOC21 & VOC20 & COCO-Object & ADE20K & Context-60 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Table 3: O / P* / BGB per dataset
rows = []
for m in M7:
    c = []
    for d in DS5:
        o, p = T[(m, "O", d)], T[(m, "P", d)]
        b = T.get((m, "BGB", d))
        c += [f2(o), f2(p), f2(b) if b is not None else "--"]
    rows.append(f"{m} & " + " & ".join(c) + r" \\")
mean = lambda s, d: np.mean([T[(m, s, d)] for m in M7])
c = []
for d in DS5:
    c += [f2(mean("O", d)), f2(mean("P", d)), f2(mean("BGB", d)) if (M7[0], "BGB", d) in T else "--"]
rows.append(r"\midrule Mean & " + " & ".join(c) + r" \\")
write("tab3_main_protocol", r"""\begin{table*}[t]
\caption{mIoU under the official configuration (O), the strict unified protocol (P*), and P* with only the official background handling transplanted (BGB). BGB is defined only for datasets with a background class.}\label{tab:main}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l|ccc|ccc|ccc|ccc|ccc@{}}
\toprule
 & \multicolumn{3}{c|}{VOC21} & \multicolumn{3}{c|}{VOC20} & \multicolumn{3}{c|}{COCO-Object} & \multicolumn{3}{c|}{ADE20K} & \multicolumn{3}{c}{Context-60} \\
Method & O & P* & BGB & O & P* & BGB & O & P* & BGB & O & P* & BGB & O & P* & BGB \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table 4: background rules (7-method means) on the three datasets with a background class
SETS = [("P", "P* (plain ``background'')"), ("OTSU", "Otsu rule"), ("SEL26", "SEL26 (26 words + Otsu)"), ("CC", "LLM vocabulary (CC)"),
        ("CF", "Conflict-filtered (CF)"), ("BGB", "Official trick (BGB)")]
rows = []
for s, lab in SETS:
    c = []
    for d in ("voc21", "coco_object", "context60"):
        vals = [T.get((m, s, d)) for m in M7]
        c.append(f2(np.mean(vals)) if all(v is not None for v in vals) else "--")
    rows.append(f"{lab} & " + " & ".join(c) + r" \\")
write("tab4_background_rules", r"""\begin{table}[t]
\caption{Background handling rules, mean mIoU over the seven methods. The tuning-free rules (Otsu, SEL26, CC, CF) use no per-dataset parameters; SEL26 was selected on training splits only. Otsu for Trident lacks its SAM refinement (corrected run: conclusions unchanged; see Appendix Table~\ref{tab:rules_full}). ``--'': not run. Per-method values are in Appendix Table~\ref{tab:rules_full}.}\label{tab:rules}
\centering\small
\begin{tabular}{@{}lccc@{}}
\toprule
Rule & VOC21 & COCO-Object & Context-60 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
# Appendix: per-method background rules
rows = []
for m in M7:
    c = []
    for d in ("voc21", "coco_object", "context60"):
        for s, _ in SETS:
            v = T.get((m, s, d)); c.append(f2(v) if v is not None else "--")
    rows.append(f"{m} & " + " & ".join(c) + r" \\")
hdr = " & ".join(["P*", "Otsu", "SEL26", "CC", "CF", "BGB"] * 3)
write("tabA1_background_rules_full", r"""\begin{table*}[t]
\caption{Per-method mIoU for the background rules of Table~\ref{tab:rules}. Otsu for Trident is the V2-P2 run without SAM refinement (corrected values 50.99/33.48 in V3; conclusions unchanged).}\label{tab:rules_full}
\centering\scriptsize\setlength{\tabcolsep}{2pt}
\begin{tabular}{@{}l|cccccc|cccccc|cccccc@{}}
\toprule
 & \multicolumn{6}{c|}{VOC21} & \multicolumn{6}{c|}{COCO-Object} & \multicolumn{6}{c}{Context-60} \\
Method & """ + hdr + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table 5: decomposition on original datasets (S3)
S3 = json.load(open(f"{ROOT}/experiments/S3/results/s3_score.json"))
rows = []
for m in M7:
    a, b = S3["voc21"]["rows"][m], S3["coco_object"]["rows"][m]
    nc = "--" if b["zero"] else sg(b["d_voc"])
    rows.append(f"{m} & {f2(a['P'])} & {f2(a['THR'])} & {f2(a['BGB'])} & {sg(a['d_thr'])} & {sg(a['d_name'])} & "
                f"{f2(b['P'])} & {f2(b['THR'])} & {f2(b['BGB'])} & {sg(b['d_thr'])} & {sg(b['d_name'])} & {nc} \\\\")
v, c = S3["voc21"], S3["coco_object"]
rows.append(r"\midrule Mean & & & & " + f"{sg(v['mean_d_thr'])} & {sg(v['mean_d_name'])} & & & & {sg(c['mean_d_thr'])} & {sg(c['mean_d_name'])} & {sg(c['mean_d_voc'])}" + r" \\")
rows.append(r"Threshold share & \multicolumn{5}{c|}{" + f"{v['share_thr']:.2f}" + r"} & \multicolumn{6}{c}{" + f"{c['share_thr']:.2f}" + r"} \\")
write("tab5_decomposition_original", r"""\begin{table*}[t]
\caption{Where the official background gain comes from on the original datasets. THR: P* plus only the numeric part of the official trick ($\tau$, $s$, $a$); BGB: THR plus the official class-name file. $\Delta_{\mathrm{thr}}=$THR$-$P*, $\Delta_{\mathrm{name}}=$BGB$-$THR; $\Delta_{\mathrm{voc}}$: gain from removing background words that collide with a query class name (``bed'' on COCO-Object; VOC21 has none). Threshold share $=\overline{\Delta_{\mathrm{thr}}}/\overline{\text{BGB}-\text{P*}}$.}\label{tab:decomp}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l|ccccc|cccccc@{}}
\toprule
 & \multicolumn{5}{c|}{VOC21} & \multicolumn{6}{c}{COCO-Object} \\
Method & P* & THR & BGB & $\Delta_{\mathrm{thr}}$ & $\Delta_{\mathrm{name}}$ & P* & THR & BGB & $\Delta_{\mathrm{thr}}$ & $\Delta_{\mathrm{name}}$ & $\Delta_{\mathrm{voc}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Table 6: decomposition on COCO-Stuff derived variants (S2)
S2 = json.load(open(f"{ROOT}/experiments/S2/results/s2_score.json"))["layers"]
col2 = json.load(open(f"{DATA}/s2r/s2/s2_collisions.json"))
rows = []
for m in M7:
    c = []
    for v in ("obj_bg30", "obj_bg50"):
        r = S2[v][m]
        c += [f2(r["P"]), f2(r["THR"]), f2(r["BGB"]), "--" if r["zero_collision"] else f2(r["NC"]), str(col2[v][m]["n_removed"]),
              "--" if r["zero_collision"] else sg(r["d_voc"])]
    rows.append(f"{m} & " + " & ".join(c) + r" \\")
write("tab6_decomposition_variants", r"""\begin{table*}[t]
\caption{Trick decomposition on COCO-Stuff variants that keep stuff classes as queries (background share 31\% and 51\%). NC: BGB with colliding background words removed; ``\#rm'': number of removed words; $\Delta_{\mathrm{voc}}=$NC$-$BGB. ClearCLIP's official file has no background words.}\label{tab:decomp_var}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}l|cccccc|cccccc@{}}
\toprule
 & \multicolumn{6}{c|}{COCO 31\% background (165 classes)} & \multicolumn{6}{c}{COCO 51\% background (151 classes)} \\
Method & P* & THR & BGB & NC & \#rm & $\Delta_{\mathrm{voc}}$ & P* & THR & BGB & NC & \#rm & $\Delta_{\mathrm{voc}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# Appendix: S1 derived variants, per method P* / BGB
S1c = {(r["method"], r["variant"], r["setting"]): float(r["mIoU"]) for r in csv.DictReader(open(f"{ROOT}/experiments/S1/results/cells_transcribed.csv"))}
VV = ["ctx_bg30", "ctx_bg50", "ctx_bg70", "obj_bg30", "obj_bg50"]
rows = [f"{m} & " + " & ".join(f"{f2(S1c[(m, v, 'P')])}/{f2(S1c[(m, v, 'BGB')])}" for v in VV) + r" \\" for m in M7]
write("tabA2_variants_full", r"""\begin{table}[t]
\caption{Derived variants with controlled background share: P*/BGB per method.}\label{tab:variants_full}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccc@{}}
\toprule
Method & Ctx 35\% & Ctx 53\% & Ctx 71\% & COCO 31\% & COCO 51\% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print("tables written to", OUT)
