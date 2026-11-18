import json, numpy as np
M7 = ["SCLIP","ClearCLIP","NACLIP","ResCLIP","SC-CLIP","ProxyCLIP","Trident"]
cls = [l.split(",")[0].strip() for l in open("/root/autodl-tmp/p2/repos/ClearCLIP/configs/cls_voc21.txt").read().split("\n") if l.strip()]
A = json.load(open("/root/autodl-tmp/s5/A.json")); K = A["K_idx"]
def iou(p): return np.array([np.nan if v is None else 100*v for v in json.load(open(p))["iou"]])
def miou(p): return json.load(open(p))["mIoU"]
S = np.array([iou(f"/root/autodl-tmp/p3/test/{m}__voc21__SEL.json") for m in M7])
C = np.array([iou(f"/root/autodl-tmp/p4/test/{m}__voc21__CC.json") for m in M7])
N = np.array([iou(f"/root/autodl-tmp/s5/stage2/{m}__voc21__CCminusA.json") for m in M7])
print("mIoU SEL", [round(miou(f"/root/autodl-tmp/p3/test/{m}__voc21__SEL.json"),2) for m in M7])
print("mIoU CC ", [round(miou(f"/root/autodl-tmp/p4/test/{m}__voc21__CC.json"),2) for m in M7])
print("mIoU CC-A", [round(miou(f"/root/autodl-tmp/s5/stage2/{m}__voc21__CCminusA.json"),2) for m in M7])
s, c, n = S.mean(0), C.mean(0), N.mean(0)
D = s - c; R = n - c
for k in K: print(cls[k], "D", round(D[k],2), "R", round(R[k],2))
F = R[K].sum() / D[K].sum()
print("F", round(F,3), "sumR", round(R[K].sum(),2), "sumD", round(D[K].sum(),2))
others = [k for k in range(len(cls)) if k not in K]
print("others mean change", round(np.nanmean(R[others]),2), "min", round(np.nanmin(R[others]),2), cls[others[int(np.nanargmin(R[others]))]], "max", round(np.nanmax(R[others]),2), cls[others[int(np.nanargmax(R[others]))]])
print("7-mean mIoU SEL CC CC-A", round(np.nanmean(s),2), round(np.nanmean(c),2), round(np.nanmean(n),2))
