"""S4 deviation 6 (B): like s4_render.py, plus intermediate maps of the official post-processing, recomputed from the
raw logits of the SAME forward pass (the official postprocess_result output is untouched and is what gets evaluated).
Saved per image (npz): P (class probabilities after softmax + max over names) for background and GT classes (float16),
argmax before thresholds, argmax after the area threshold, confidence (max P), and the recomputed final prediction.
Consistency gate: recomputed final == official pred_sem_seg, pixel for pixel (recorded as 'consistent').
Usage: python s4b_render.py SCRIPT METHOD SETTING DATASET OUT_JSON IDX[,IDX...] OUTDIR"""
import json
import os
import runpy
import sys

script = sys.argv.pop(1)
idxs = [int(x) for x in sys.argv.pop(5).split(",")]
outdir = os.path.abspath(sys.argv.pop(5))
os.makedirs(outdir, exist_ok=True)
tag = "__".join(sys.argv[1:4])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import mmengine.runner  # noqa: E402
from mmseg.evaluation import IoUMetric  # noqa: E402
from mmseg.registry import MODELS  # noqa: E402

_count = {"n": 0, "pp": 0}
_log, _inter = [], {}


def wrap_postprocess(cls):
    orig = cls.postprocess_result
    if getattr(orig, "_s4b", False):
        return

    def pp(self, seg_logits, data_samples):
        raw = seg_logits.detach().clone()
        out = orig(self, seg_logits, data_samples)
        with torch.no_grad():
            x = (raw[0] * self.logit_scale).softmax(0)
            qi = self.query_idx
            num_cls = int(max(qi)) + 1
            if num_cls != len(qi):
                ci = torch.nn.functional.one_hot(qi).T.view(num_cls, len(qi), 1, 1).to(x.dtype)
                P = torch.stack([(x * ci[c]).max(0)[0] for c in range(num_cls)], 0)
            else:
                P = x
            arg0 = P.argmax(0)
            Pa = P.clone()
            removed = []
            if self.area_thd is not None:
                pr = torch.nn.functional.one_hot(Pa.argmax(0), num_cls).to(Pa.dtype)
                ap = pr[:, :, 1:].sum((0, 1), keepdim=True)
                keep = (ap > self.area_thd * ap.sum()).to(Pa.dtype)
                removed = [int(c) + 1 for c in torch.nonzero((keep.flatten() == 0) & (ap.flatten() > 0)).flatten()]
                Pa[1:] *= keep.transpose(0, -1)
            arg1 = Pa.argmax(0)
            conf = Pa.max(0)[0]
            fin = arg1.clone()
            fin[conf < self.prob_thd] = 0
            off = out[0].pred_sem_seg.data.squeeze(0)
            _inter[_count["pp"]] = {"P": P.half().cpu().numpy(), "argmax_raw": arg0.cpu().numpy().astype(np.uint8),
                                     "argmax_area": arg1.cpu().numpy().astype(np.uint8),
                                     "conf": conf.half().cpu().numpy(), "final_recomputed": fin.cpu().numpy().astype(np.uint8),
                                     "consistent": bool(torch.equal(fin, off)), "removed_by_area": removed,
                                     "logit_scale": float(self.logit_scale), "prob_thd": float(self.prob_thd),
                                     "area_thd": None if self.area_thd is None else float(self.area_thd)}
            _count["pp"] += 1
        return out

    pp._s4b = True
    cls.postprocess_result = pp


_from_cfg = mmengine.runner.Runner.from_cfg.__func__


def from_cfg(cls, cfg):
    cfg.test_dataloader.dataset.indices = idxs
    cfg.test_dataloader.batch_size = 1
    wrap_postprocess(MODELS.get(cfg.model["type"]))
    return _from_cfg(cls, cfg)


mmengine.runner.Runner.from_cfg = classmethod(from_cfg)

_process = IoUMetric.process


def process(self, data_batch, data_samples):
    for ds in data_samples:
        k = _count["n"]
        i = idxs[k]
        pred = ds["pred_sem_seg"]["data"].squeeze().cpu().numpy().astype(np.uint8)
        gt = ds["gt_sem_seg"]["data"].squeeze().cpu().numpy().astype(np.uint8)
        Image.fromarray(pred).save(f"{outdir}/{tag}__{i}__pred.png")
        Image.fromarray(gt).save(f"{outdir}/{tag}__{i}__gt.png")
        it = _inter.pop(k)
        keep = sorted({0} | {int(c) for c in np.unique(gt) if c != 255})
        np.savez_compressed(f"{outdir}/{tag}__{i}__inter.npz", P_classes=np.array(keep),
                            P=it["P"][keep], argmax_raw=it["argmax_raw"], argmax_area=it["argmax_area"],
                            conf=it["conf"], final_recomputed=it["final_recomputed"])
        _log.append({"index": i, "img_path": ds["img_path"], "seg_map_path": ds.get("seg_map_path"),
                     "shape": list(pred.shape), "consistent": it["consistent"] and bool(np.array_equal(it["final_recomputed"], pred)),
                     "removed_by_area": it["removed_by_area"], "logit_scale": it["logit_scale"],
                     "prob_thd": it["prob_thd"], "area_thd": it["area_thd"]})
        _count["n"] += 1
    return _process(self, data_batch, data_samples)


IoUMetric.process = process

sys.argv = [script] + sys.argv[1:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    json.dump({"tag": tag, "indices": idxs, "images": _log}, open(f"{outdir}/{tag}__paths.json", "w"), indent=1)
    print("S4B_RENDERED", tag, len(_log), "consistent", all(x["consistent"] for x in _log))
