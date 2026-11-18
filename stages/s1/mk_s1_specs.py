"""S1: build per-variant specs and query-name files from the frozen Tier 1 merge lists (tier1_bgfrac.json)."""
import json
import os

from mmengine.config import Config

A = "/root/autodl-tmp"
S1 = f"{A}/s1"
R = f"{A}/p2/repos"
U = f"{R}/ClearCLIP/configs"
M7 = ["SCLIP", "ClearCLIP", "NACLIP", "ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]


def lines(p):
    return [l.rstrip("\r") for l in open(p).read().split("\n") if l.strip()]


def official_names(m, parent):
    cfg = Config.fromfile(f"{R}/{m}/configs/cfg_{parent}.py")
    p = cfg.model["name_path"]
    return lines(p if os.path.isabs(p) else os.path.normpath(os.path.join(f"{R}/{m}", p)))


rep = json.load(open(f"{S1}/tier1_bgfrac.json"))
os.makedirs(f"{S1}/specs", exist_ok=True)
os.makedirs(f"{S1}/names", exist_ok=True)
uctx = lines(f"{U}/cls_context60.txt"); uobj = lines(f"{U}/cls_coco_object.txt"); ustuff = lines(f"{U}/cls_coco_stuff.txt")
assert len(uctx) == 60 and len(uobj) == 81 and len(ustuff) == 171
for k, v in rep["ctx_variants"].items():
    keep = [0] + v["keep"]
    spec = {"parent": "context60", "data_root": f"{S1}/data/{k}", "classes": [uctx[c] for c in keep], "names_BGB": {}}
    spec["names_P"] = f"{S1}/names/{k}__P.txt"
    open(spec["names_P"], "w").write("\n".join(uctx[c] for c in keep) + "\n")
    for m in M7:
        on = official_names(m, "context60")
        assert len(on) == 60, (m, len(on))
        p = f"{S1}/names/{k}__BGB__{m}.txt"
        open(p, "w").write("\n".join(on[c] for c in keep) + "\n")
        spec["names_BGB"][m] = p
    json.dump(spec, open(f"{S1}/specs/{k}.json", "w"), indent=1)
    print(k, len(spec["classes"]))
for k, v in rep["obj_variants"].items():
    extra = [ustuff[80 + (c - 91)] for c in v["keep"]]
    spec = {"parent": "coco_object", "data_root": f"{S1}/data/{k}", "classes": uobj + extra, "names_BGB": {}}
    spec["names_P"] = f"{S1}/names/{k}__P.txt"
    open(spec["names_P"], "w").write("\n".join(uobj + extra) + "\n")
    for m in M7:
        on = official_names(m, "coco_object")
        assert len(on) == 81, (m, len(on))
        p = f"{S1}/names/{k}__BGB__{m}.txt"
        open(p, "w").write("\n".join(on + extra) + "\n")
        spec["names_BGB"][m] = p
    assert len(set(spec["classes"])) == len(spec["classes"])
    json.dump(spec, open(f"{S1}/specs/{k}.json", "w"), indent=1)
    print(k, len(spec["classes"]))
