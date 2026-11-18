import csv, json, glob, os, sys
I = os.path.expanduser("~/mnt/OVSS_Journal_20261107/inputs")
out = []
# p2 table
raw = {}
for f in glob.glob("p2_results_core/results/full/*.json"):
    r = json.load(open(f))
    if "mIoU" in r and "method" in r:
        raw[(r["method"], r["setting"], r["dataset"])] = r["mIoU"]
rows = list(csv.DictReader(open(f"{I}/Probe_OVSSProtocol02/full_score/p2_table_transcribed.csv")))
bad = []
for r in rows:
    k = (r["method"], r["setting"], r["dataset"])
    v = raw.get(k)
    if v is None or abs(round(v, 2) - float(r["mIoU"])) > 0.005:
        bad.append((k, r["mIoU"], v))
print("p2: rows", len(rows), "raw cells", len(raw), "mismatch", len(bad)); [print("  ", b) for b in bad]
# p5 cells
raw5 = {}
for f in glob.glob("p5_results/p5/cells/*.json"):
    r = json.load(open(f)); b = os.path.basename(f)[:-5].split("__")
    raw5[tuple(b)] = r["mIoU"]
print("p5 raw cells", len(raw5), sorted({k[1] for k in raw5}))
rows5 = list(csv.DictReader(open(f"{I}/Probe_OVSSContext05/results/p5_cells_transcribed.csv")))
bad5 = []
p2s = {("BGB"): "BGB", "P": "P"}
for r in rows5:
    m, d = r["method"], r["dataset"]
    for col in ["CF", "BGB", "P", "SEL26", "CC"]:
        if not r[col]:
            continue
        x = float(r[col])
        if d == "context60":
            v = raw5.get((m, col, d))
        elif col == "CF":
            v = raw5.get((m, col, d))
        elif col in ("BGB", "P"):
            v = raw.get((m, col, d))
        else:
            v = "skip"  # SEL26/CC on voc/coco come from V3/V4 json (not in these packages)
        if v == "skip":
            continue
        if v is None or abs(round(v, 2) - x) > 0.005:
            bad5.append((m, d, col, x, v))
print("p5: rows", len(rows5), "mismatch", len(bad5)); [print("  ", b) for b in bad5]
# p5 O column for context (RESULT table) and G0
for m in ["SCLIP","ClearCLIP","NACLIP","ResCLIP","SC-CLIP","ProxyCLIP","Trident"]:
    print("ctx", m, "O", round(raw5[(m,"O","context60")],2), "P", round(raw5[(m,"P","context60")],2), "BGB", round(raw5[(m,"BGB","context60")],2), "SEL26", round(raw5[(m,"SEL26","context60")],2))
