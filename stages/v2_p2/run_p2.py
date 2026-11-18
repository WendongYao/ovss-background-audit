"""Orchestrate all V2-P2 cells; resumable (skips cells with a result file) and pausable (touch PAUSE).
Usage: python run_p2.py smoke|full [METHOD ...]
Order: gate cells (O) for the new methods first, then P* for all, then the background cells (BGB, OTSU)."""

import json
import os
import subprocess
import sys
import time

W = "/root/autodl-tmp/p2"
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = ["ResCLIP", "SC-CLIP", "ProxyCLIP", "Trident"]
OLD = ["SCLIP", "ClearCLIP", "NACLIP"]
DATASETS = ["voc21", "voc20", "coco_object", "ade20k"]
BG_DATASETS = ["voc21", "coco_object"]


def cells(methods):
    out = []
    for m in [x for x in NEW + OLD if x in methods]:
        out += [(m, "O", d) for d in DATASETS]
    for m in [x for x in NEW + OLD if x in methods]:
        out += [(m, "P", d) for d in DATASETS]
    for m in [x for x in NEW + OLD if x in methods]:
        out += [(m, s, d) for s in ("BGB", "OTSU") for d in BG_DATASETS]
    return out


def main():
    mode = sys.argv[1]
    assert mode in ("smoke", "full")
    methods = sys.argv[2:] or NEW + OLD
    res_dir = f"{W}/results/{mode}"
    os.makedirs(res_dir, exist_ok=True)
    status = []
    t0 = time.time()
    for m, s, d in cells(methods):
        if os.path.exists(f"{W}/PAUSE"):
            print(f"[{time.strftime('%H:%M:%S')}] PAUSED (remove {W}/PAUSE and rerun to continue)", flush=True)
            return
        out = f"{res_dir}/{m}__{s}__{d}.json"
        if os.path.exists(out):
            continue
        log = out.replace(".json", ".log")
        cmd = [sys.executable, f"{HERE}/run_one.py", m, s, d, out] + (["--smoke", "20"] if mode == "smoke" else [])
        tc = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        ok = rc == 0 and os.path.exists(out)
        miou = json.load(open(out))["mIoU"] if ok else None
        status.append({"method": m, "setting": s, "dataset": d, "ok": ok, "rc": rc, "mIoU": miou, "seconds": time.time() - tc})
        print(f"[{time.strftime('%H:%M:%S')}] {m:9s} {s:4s} {d:12s} {'OK ' + format(miou, '.2f') if ok else 'FAILED rc=' + str(rc)}"
              f"  {time.time() - tc:.0f}s  total {time.time() - t0:.0f}s", flush=True)
        json.dump(status, open(f"{res_dir}/status_{int(t0)}.json", "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] ALL CELLS DONE ({mode})", flush=True)


if __name__ == "__main__":
    main()
