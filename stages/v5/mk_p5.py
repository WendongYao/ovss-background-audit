"""Derive p5_run.py (from V4 p4_run.py) and p5_run_one.py (from phase-2 run_one.py): only add Context-60 data roots."""
import hashlib
import sys

SRC4 = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/p4/code/p4_run.py"
SRC2 = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/p2/code/run_one.py"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/root/autodl-tmp/p5/code"
for p, h in [(SRC4, "8c4aabbb8b6d71c988cf4cbd1c0f4868"), (SRC2, "e1ced23ad6ce1736a8809a3985ba8470")]:
    assert hashlib.md5(open(p, "rb").read()).hexdigest() == h, p

s = open(SRC4).read()
old = '        ("coco_object", "val"): f"{W2}/data/coco_object", ("coco_object", "dev"): f"{W2}/data/coco_object_dev"}'
assert s.count(old) == 1
s = s.replace(old, old[:-1] + ',\n        ("context60", "val"): f"{W2}/data/VOCdevkit/VOC2010"}')
open(f"{OUT}/p5_run.py", "w").write(s)

t = open(SRC2).read()
old = '        "coco_object": f"{W}/data/coco_object", "ade20k": f"{W}/data/ade/ADEChallengeData2016"}'
assert t.count(old) == 1
t = t.replace(old, old[:-1] + ',\n        "context60": f"{W}/data/VOCdevkit/VOC2010"}')
open(f"{OUT}/p5_run_one.py", "w").write(t)
print("ok", hashlib.md5(s.encode()).hexdigest(), hashlib.md5(t.encode()).hexdigest())
