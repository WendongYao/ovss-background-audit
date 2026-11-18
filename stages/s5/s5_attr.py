"""S5 layer 3: attribution set A from stage-1 dev LOO results (rule frozen in CONTRACT_S5_CN.md).
A_c = {w : 6-method mean dev IoU_c(CC - w) - IoU_c(CC) >= 1.0 (IoU points)}, c in K; A = union.
Usage: python s5_attr.py STAGE1_DIR WORDS_JSON CLS_FILE OUT_A_JSON OUT_WORDS_JSON"""
import json
import sys

import numpy as np

S1, WJ, CF, OA, OW = sys.argv[1:6]
M6 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP"]
K_NAMES = ["ship", "tvmonitor", "table", "person", "bottle", "bird", "train"]
words = json.load(open(WJ))["voc21"]
cls = [l.split(",")[0].strip() for l in open(CF).read().split("\n") if l.strip()]
K = [cls.index(n) for n in K_NAMES]
gain = []
for m in M6:
    d = np.load(f"{S1}/{m}__voc21__dev_loo.npz")
    var = list(d["variant"])
    assert int(d["replica_mismatch"]) == 0, (m, int(d["replica_mismatch"]))
    assert var == [-1] + list(range(len(words))), (m, len(var))
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = 100 * d["intersect"] / d["union"]
    gain.append(iou[1:] - iou[0])          # [n_words, n_cls]
g = np.mean(gain, 0)
A_c = {cls[c]: [words[j] for j in range(len(words)) if g[j, c] >= 1.0] for c in K}
A = sorted({w for v in A_c.values() for w in v}, key=words.index)
table = {words[j]: {cls[c]: round(float(g[j, c]), 3) for c in K} for j in range(len(words))}
json.dump({"K": K_NAMES, "K_idx": K, "A_c": A_c, "A": A, "gain_6method_mean_dev": table}, open(OA, "w"), indent=1,
          ensure_ascii=False)
json.dump({"voc21": [w for w in words if w not in A]}, open(OW, "w"), indent=1, ensure_ascii=False)
print("S5_ATTR K", K, "A", A, "n_left", len(words) - len(A))
for c, v in A_c.items():
    print(" ", c, v)
