"""S3: specs for the original VOC21 / COCO-Object validation sets and NC name files (S2 collision rule, verbatim)."""
import hashlib
import json
import os

from mmengine.config import Config

A = "/root/autodl-tmp"; R = f"{A}/p2/repos"; U = f"{R}/ClearCLIP/configs"; S3 = f"{A}/s3"
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
SEPC = {"ProxyCLIP": ";", "Trident": ";"}
JOIN = {"ProxyCLIP": "; ", "Trident": "; "}
DATA = {"voc21": f"{A}/p2/data/VOCdevkit/VOC2012", "coco_object": f"{A}/p2/data/coco_object"}  # as V2-P2 run_one.py


def norm(s):
    p = s.lower().replace("-", " ").replace("_", " ").split()
    if p and p[-1] in ("other", "merged"):
        p = p[:-1]
    return " ".join(p)


def collide(w, q):
    w, q = norm(w), norm(q)
    if not w or not q:
        return False
    return w == q or w in q.split() or q in w.split()


def lines(p):
    return [l.rstrip("\r") for l in open(p).read().split("\n") if l.strip()]


os.makedirs(f"{S3}/specs", exist_ok=True); os.makedirs(f"{S3}/names", exist_ok=True)
col = {}
for ds in ["voc21", "coco_object"]:
    v = f"{ds}_orig"
    spec = {"parent": ds, "data_root": DATA[ds], "classes": lines(f"{U}/cls_{ds}.txt"), "names_P": f"{U}/cls_{ds}.txt", "names_BGB": {}}
    col[v] = {}
    for m in M7:
        cfg = Config.fromfile(f"{R}/{m}/configs/cfg_{ds}.py")
        p = cfg.model["name_path"]; p = p if os.path.isabs(p) else os.path.normpath(os.path.join(f"{R}/{m}", p))
        spec["names_BGB"][m] = p
        L = lines(p); assert len(L) == len(spec["classes"]), (m, ds, len(L))
        c = SEPC.get(m, ",")
        bg = [t.strip() for t in L[0].split(c) if t.strip()]
        q = [t.strip() for l in L[1:] for t in l.split(c) if t.strip()]
        removed = [w for w in bg if w.lower() != "background" and any(collide(w, x) for x in q)]
        hits = {w: sorted({x for x in q if collide(w, x)}) for w in removed}
        kept = [w for w in bg if w not in removed] or ["background"]
        nc = f"{S3}/names/{v}__NC__{m}.txt"
        open(nc, "w").write("\n".join([JOIN.get(m, ", ").join(kept)] + L[1:]) + "\n")
        col[v][m] = {"n_bg_words": len(bg), "n_removed": len(removed), "removed": hits, "zero": len(removed) == 0, "nc_path": nc, "bgb_path": p}
        print(v, m, len(bg), len(removed), hits)
    json.dump(spec, open(f"{S3}/specs/{v}.json", "w"), indent=1)
json.dump(col, open(f"{S3}/s3_collisions.json", "w"), indent=1)
if not os.path.exists(f"{S3}/s2_collisions.json"):
    os.symlink(f"{S3}/s3_collisions.json", f"{S3}/s2_collisions.json")  # name expected by s2_run_one.main
print("sha256", hashlib.sha256(open(f"{S3}/s3_collisions.json", "rb").read()).hexdigest())
