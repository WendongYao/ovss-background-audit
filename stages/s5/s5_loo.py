"""S5 layer 3, stage 1: leave-one-word-out on the V4 CC vocabulary, in ONE forward pass per method.
Wraps the UNCHANGED V4 p4_run.py (test mode). For every image, after the backbone has produced the raw similarities:
  - the genuine V4 post-processing (p4's own wrapped postprocess_result) runs as usual and is what IoUMetric evaluates;
  - a replica of the same V4 logic is evaluated for the full CC list and for each 'CC minus word j' (j = 0..n-1),
    and per-class intersect / union are accumulated per variant.
Check: the replica's full-CC prediction must equal the genuine prediction on every image (counted in 'replica_mismatch').
Usage: python s5_loo.py P4_RUN METHOD test DATASET SPLIT OUT_JSON --rule RULE --bgwords WORDS [--smoke N]
Env S5_LOO=0 -> no variants (only the genuine run + replica check)."""
import os
import runpy
import sys

import numpy as np

p4 = sys.argv.pop(1)
LOO = os.environ.get("S5_LOO", "1") != "0"
out = os.path.abspath(sys.argv[5])

import torch  # noqa: E402
import mmengine.runner  # noqa: E402
from mmseg.registry import MODELS  # noqa: E402

ACC = {}
STATE = {"img": 0, "mismatch": 0}


def confusion(pred, gt, n):
    m = gt != 255
    idx = gt[m].long() * n + pred[m].long()
    cm = torch.bincount(idx, minlength=n * n).view(n, n).double()
    inter = cm.diag()
    return inter, cm.sum(0) + cm.sum(1) - inter


def install(cls, g):
    p4w = cls.postprocess_result
    fv = dict(zip(p4w.__code__.co_freevars, [c.cell_contents for c in p4w.__closure__]))
    orig, rule, nbg_all, n_cls = fv["orig"], fv["rule"], fv["nbg_all"], fv["n_cls"]
    decide = g["decide"]
    n_words = nbg_all - 1

    def replica(self, seg_logits, data_samples, drop_word):
        if drop_word is None:
            x, nba = seg_logits, nbg_all
        else:
            r = 1 + drop_word
            x, nba = torch.cat([seg_logits[:, :r], seg_logits[:, r + 1:]], 1), nbg_all - 1
        new = x.clone()
        for i in range(x.shape[0]):
            bg, keep = decide(x[i], nba, rule, torch)
            nb = 1 if rule["V"] == 0 else nba
            new[i, :nba] = -10.0
            new[i, :nb] = torch.where(bg, 10.0, -10.0).to(new.dtype)
            drop = torch.nonzero(~keep).flatten() + nba
            if len(drop):
                new[i, drop] = -10.0
        small = torch.cat([new[:, :1], new[:, nba:]], 1)
        qi_orig = self.query_idx
        self.query_idx = torch.arange(n_cls, device=qi_orig.device, dtype=qi_orig.dtype) if hasattr(qi_orig, "device") \
            else list(range(n_cls))
        try:
            orig(self, small, data_samples)
        finally:
            self.query_idx = qi_orig
        return [d.pred_sem_seg.data[0].clone() for d in data_samples]

    def pp(self, seg_logits, data_samples, *a, **k):
        raw = seg_logits.detach().clone()
        variants = ([None] + list(range(n_words))) if LOO else [None]
        for v in variants:
            preds = replica(self, raw, data_samples, v)
            key = -1 if v is None else v
            if key == -1:
                rep_full = preds
            for i, p in enumerate(preds):
                gt = data_samples[i].gt_sem_seg.data[0].to(p.device)
                it, un = confusion(p, gt, n_cls)
                a_i, a_u = ACC.setdefault(key, [torch.zeros(n_cls, dtype=torch.float64, device=p.device),
                                                torch.zeros(n_cls, dtype=torch.float64, device=p.device)])
                a_i += it; a_u += un
        res = p4w(self, seg_logits, data_samples, *a, **k)
        for i, d in enumerate(res):
            if not torch.equal(d.pred_sem_seg.data[0], rep_full[i]):
                STATE["mismatch"] += 1
        STATE["img"] += len(res)
        return res

    cls.postprocess_result = pp
    STATE["n_words"] = n_words
    STATE["n_cls"] = n_cls


_from_cfg = mmengine.runner.Runner.from_cfg.__func__
G = {}


def from_cfg(c, cfg):
    install(MODELS.get(cfg.model["type"]), G["g"])
    return _from_cfg(c, cfg)


mmengine.runner.Runner.from_cfg = classmethod(from_cfg)

sys.argv = [p4] + sys.argv[1:]
src = open(p4).read().replace('if __name__ == "__main__":\n    main()', "")
G["g"] = {"__name__": "p4_run_s5", "__file__": p4}
exec(compile(src, p4, "exec"), G["g"])
try:
    G["g"]["main"]()
finally:
    keys = sorted(ACC)
    np.savez_compressed(out.replace(".json", "_loo.npz"), variant=np.array(keys),
                        intersect=np.stack([ACC[k][0].cpu().numpy() for k in keys]) if keys else np.zeros(0),
                        union=np.stack([ACC[k][1].cpu().numpy() for k in keys]) if keys else np.zeros(0),
                        n_img=STATE["img"], replica_mismatch=STATE["mismatch"])
    print("S5_LOO", os.path.basename(out), "images", STATE["img"], "variants", len(keys),
          "replica_mismatch", STATE["mismatch"])
