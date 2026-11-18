"""V5 data: PASCAL Context-60 val labels from PyTorch-Encoding's val.pth (dict image-id -> mask, 0 = background,
1..59 classes). The pickle is loaded with a restricted unpickler that only allows PIL / numpy / torch rebuild globals.
Writes VOCdevkit/VOC2010/{SegmentationClassContext/*.png, ImageSets/SegmentationContext/val.txt, JPEGImages -> VOC2012}."""

import json
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image

SRC = "/root/autodl-tmp/p2/data/raw/pcontext/val.pth"
VOC12 = "/root/autodl-tmp/p2/data/VOCdevkit/VOC2012"
OUT = "/root/autodl-tmp/p2/data/VOCdevkit/VOC2010"
REPORT = "/root/autodl-tmp/p5/ctxdata_report.json"
ALLOWED_MODULES = ("PIL.", "numpy", "collections", "copyreg", "torch._utils", "torch")
ALLOWED_NAMES = {("builtins", "set"), ("builtins", "dict"), ("builtins", "list"), ("builtins", "tuple"), ("builtins", "bytes"),
                 ("_codecs", "encode")}
seen = set()


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        seen.add((module, name))
        ok = (module, name) in ALLOWED_NAMES or (module.startswith(ALLOWED_MODULES) and not name.startswith("_load")
                                               and name not in ("load", "loads", "eval", "exec", "system"))
        if module == "torch" and not (name.endswith("Storage") or name in ("Size", "device", "float32", "uint8", "int64")):
            ok = False
        if not ok:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


class SafePickle:
    Unpickler = SafeUnpickler
    load = staticmethod(lambda f, **kw: SafeUnpickler(f, **kw).load())
    __name__ = "safe_pickle"


def main():
    try:
        masks = torch.load(SRC, map_location="cpu", pickle_module=SafePickle, weights_only=False)
    except Exception as e:  # noqa: BLE001
        print("LOAD FAILED", repr(e), "globals seen:", sorted(seen))
        sys.exit(2)
    print("globals seen:", sorted(seen))
    print("type", type(masks), "n", len(masks), "example key", next(iter(masks)))
    os.makedirs(f"{OUT}/SegmentationClassContext", exist_ok=True)
    os.makedirs(f"{OUT}/ImageSets/SegmentationContext", exist_ok=True)
    if not os.path.exists(f"{OUT}/JPEGImages"):
        os.symlink(f"{VOC12}/JPEGImages", f"{OUT}/JPEGImages")
    names, vmax, vmin, missing = [], 0, 255, []
    hist = np.zeros(256, np.int64)
    for k, m in masks.items():
        s = str(k)
        name = s if "_" in s else s[:4] + "_" + s[4:]
        name = name.replace(".jpg", "").replace(".png", "")
        a = np.array(m).astype(np.int64)
        vmax, vmin = max(vmax, int(a.max())), min(vmin, int(a.min()))
        hist += np.bincount(a.ravel(), minlength=256)[:256]
        if not os.path.exists(f"{VOC12}/JPEGImages/{name}.jpg"):
            missing.append(name)
            continue
        img = Image.open(f"{VOC12}/JPEGImages/{name}.jpg")
        assert img.size == (a.shape[1], a.shape[0]), (name, img.size, a.shape)
        Image.fromarray(a.astype(np.uint8)).save(f"{OUT}/SegmentationClassContext/{name}.png")
        names.append(name)
    names.sort()
    open(f"{OUT}/ImageSets/SegmentationContext/val.txt", "w").write("\n".join(names) + "\n")
    rep = {"n_masks": len(masks), "n_written": len(names), "missing_images": missing[:20], "n_missing": len(missing),
           "label_min": vmin, "label_max": vmax, "n_labels_present": int((hist > 0).sum()),
           "background_fraction": float(hist[0] / hist.sum()), "globals": sorted(f"{a}.{b}" for a, b in seen)}
    json.dump(rep, open(REPORT, "w"), indent=1)
    print(json.dumps(rep))
    assert len(names) == 5105 and vmax <= 59, "unexpected size or label range"
    print("context60 ok")


if __name__ == "__main__":
    main()
