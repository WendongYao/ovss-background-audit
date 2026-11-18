"""S2: NC (no-collision) query-name files and the collision list (contract experiments/CONTRACT_S2_CN.md).
Removes from the official background line only those background words that collide with a kept query word."""
import hashlib
import json
import os

S1 = "/root/autodl-tmp/s1"
S2 = "/root/autodl-tmp/s2"
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
SEPC = {"ProxyCLIP": ";", "Trident": ";"}  # split char (V4 p4_run.py SEP); others ","
JOIN = {"ProxyCLIP": "; ", "Trident": "; "}
VARIANTS = ["obj_bg30", "obj_bg50", "ctx_bg70"]


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


os.makedirs(f"{S2}/names", exist_ok=True)
out = {}
for v in VARIANTS:
    spec = json.load(open(f"{S1}/specs/{v}.json"))
    out[v] = {}
    for m in M7:
        L = lines(spec["names_BGB"][m])
        c = SEPC.get(m, ",")
        bg = [t.strip() for t in L[0].split(c) if t.strip()]
        q = [t.strip() for l in L[1:] for t in l.split(c) if t.strip()]
        removed = [w for w in bg if w.lower() != "background" and any(collide(w, x) for x in q)]
        hits = {w: sorted({x for x in q if collide(w, x)}) for w in removed}
        kept = [w for w in bg if w not in removed] or ["background"]
        p = f"{S2}/names/{v}__NC__{m}.txt"
        open(p, "w").write("\n".join([JOIN.get(m, ", ").join(kept)] + L[1:]) + "\n")
        out[v][m] = {"n_bg_words": len(bg), "n_removed": len(removed), "removed": hits, "zero": len(removed) == 0,
                     "nc_path": p, "bgb_path": spec["names_BGB"][m]}
        print(v, m, len(bg), len(removed), sorted(removed)[:15])
json.dump(out, open(f"{S2}/s2_collisions.json", "w"), indent=1)
print("sha256", hashlib.sha256(open(f"{S2}/s2_collisions.json", "rb").read()).hexdigest())
