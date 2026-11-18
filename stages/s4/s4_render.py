"""S4: render predictions for the pre-selected images with the SAME scripts that produced the frozen per-image results.
O, P, BGB -> V2-P2 run_one.py ; THR, NC -> S3 s3_run_one.py. The only changes are injected here:
  (1) the test dataset is restricted to the selected indices (order kept);
  (2) IoUMetric.process additionally saves the predicted and GT label maps (uint8 PNG) and the image path.
Usage: python s4_render.py SCRIPT METHOD SETTING DATASET OUT_JSON IDX[,IDX...] OUTDIR
       (SCRIPT = path to run_one.py or s3_run_one.py; the per-image npz is written next to OUT_JSON as usual)"""
import json
import os
import runpy
import sys

script = sys.argv.pop(1)
idxs = [int(x) for x in sys.argv.pop(5).split(",")]
outdir = os.path.abspath(sys.argv.pop(5))
os.makedirs(outdir, exist_ok=True)
tag = "__".join(sys.argv[1:4])  # METHOD__SETTING__DATASET

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import mmengine.runner  # noqa: E402
from mmseg.evaluation import IoUMetric  # noqa: E402

_from_cfg = mmengine.runner.Runner.from_cfg.__func__


def from_cfg(cls, cfg):
    cfg.test_dataloader.dataset.indices = idxs
    cfg.test_dataloader.batch_size = 1
    return _from_cfg(cls, cfg)


mmengine.runner.Runner.from_cfg = classmethod(from_cfg)

_process = IoUMetric.process
_count = {"n": 0}
_log = []


def process(self, data_batch, data_samples):
    for ds in data_samples:
        k = _count["n"]
        i = idxs[k]
        pred = ds["pred_sem_seg"]["data"].squeeze().cpu().numpy().astype(np.uint8)
        gt = ds["gt_sem_seg"]["data"].squeeze().cpu().numpy().astype(np.uint8)
        Image.fromarray(pred).save(f"{outdir}/{tag}__{i}__pred.png")
        Image.fromarray(gt).save(f"{outdir}/{tag}__{i}__gt.png")
        _log.append({"index": i, "img_path": ds["img_path"], "seg_map_path": ds.get("seg_map_path"),
                     "shape": list(pred.shape)})
        _count["n"] += 1
    return _process(self, data_batch, data_samples)


IoUMetric.process = process

sys.argv = [script] + sys.argv[1:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    json.dump({"tag": tag, "indices": idxs, "images": _log}, open(f"{outdir}/{tag}__paths.json", "w"), indent=1)
    print("S4_RENDERED", tag, len(_log))
