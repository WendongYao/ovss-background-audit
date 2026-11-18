"""LaTeX table from data/literature_audit.csv (literature/config audit; evidence in notes/literature_audit_CN.md)."""
import csv, sys
src, out = sys.argv[1:3]
SYM = {"yes": r"\ding{51}", "no": r"\ding{55}", "partial": r"$\circ$", "n/a": "--", "none": "--", "minor": "minor", "opt": "opt.", "unverified": "?"}
rows = []
for r in csv.DictReader(open(src)):
    s = lambda k: SYM.get(r[k], r[k])
    bg = "--" if "n/a" in (r["bgw_voc21"], r["bgw_cocoobj"]) else f"{r['bgw_voc21']}/{r['bgw_cocoobj']}"
    rows.append(f"{r['method']} & {r['venue']} & {r['thr']} & {bg} & {s('rename')} & {s('post')} & {s('res')} & {s('doc')} & {s('untuned')} \\\\")
open(out, "w").write(r"""\begin{table}[pos=!htbp]
\caption{Literature and configuration audit of training-free OVSS methods (official repositories and paper text, checked November 2026). Thr.: per-dataset background decision in the official configs (VOC21/COCO-Object/Context-60); \#bg: background words in the official class-name file (VOC21/COCO-Object); Ren.: class renaming or synonyms; Post.: pixel-level post-processing in the main table; Res.: resolution or window differs across datasets; Doc.: the paper text documents the background choices; Untuned: results without the dataset-specific choices are reported. \ding{51}~yes, $\circ$~partial, \ding{55}~no, ?~not verifiable from public material, --~not applicable.}\label{tab:litaudit}
\centering\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lllcccccc@{}}
\toprule
Method & Venue & Thr. & \#bg & Ren. & Post. & Res. & Doc. & Untuned \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print("ok")
