"""V4 (based on V3 p3_run.py; background word list configurable via --bgwords, background queries merged before the
method's own post-processing). V3: cache per-query similarities (dev) or apply a background rule (test) on top of each method's own post-processing.
Usage: python p3_run.py METHOD MODE DATASET SPLIT OUT_JSON [--rule RULE_JSON] [--smoke N]
MODE: cache | test ; DATASET: voc21 | coco_object ; SPLIT: dev | val
Settings are the phase-2 P* settings, except the query file = unified class names + 26 background words
(as background synonyms on the first line), as frozen in CONTRACT_CN.md."""

import inspect
import json
import os
import sys
import time

import numpy as np

W2 = "/root/autodl-tmp/p2"
W3 = "/root/autodl-tmp/p3"
MODULES = {"SCLIP": "clip_segmentor", "ClearCLIP": "clearclip_segmentor", "NACLIP": "naclip",
           "ResCLIP": "resclip_segmentor", "SC-CLIP": "scclip_segmentor", "ProxyCLIP": "proxyclip_segmentor",
           "Trident": "trident"}
SEP = {"ProxyCLIP": "; ", "Trident": "; "}  # these two repos split synonyms on '; ', the others on ', ' / ','
DATA = {("voc21", "val"): f"{W2}/data/VOCdevkit/VOC2012", ("voc21", "dev"): f"{W2}/data/VOCdevkit/VOC2012",
        ("coco_object", "val"): f"{W2}/data/coco_object", ("coco_object", "dev"): f"{W2}/data/coco_object_dev"}
BG26 = ["sky", "wall", "tree", "wood", "grass", "road", "sea", "river", "mountain", "sands", "desk", "bed", "building",
        "cloud", "lamp", "door", "window", "wardrobe", "ceiling", "shelf", "curtain", "stair", "floor", "hill", "rail",
        "fence"]
P_SHORT = 448
DS = 4  # cache downsampling factor
TAG = "v1"


def name_file(method, ds, bgw):
    lines = [l.strip() for l in open(f"{W2}/repos/ClearCLIP/configs/cls_{ds}.txt").read().split("\n") if l.strip()]
    assert lines[0] == "background", lines[0]
    sep = SEP.get(method, ", ")
    os.makedirs(f"{W3}/names", exist_ok=True)
    p = f"{W3}/names/cls_{ds}_{TAG}_{method}.txt"
    open(p, "w").write("\n".join([sep.join(["background"] + bgw)] + lines[1:]) + "\n")
    return p, len(lines)


def otsu_thr(fg, torch):
    hist = torch.histc(fg, bins=256, min=0, max=1)
    p = hist / hist.sum()
    w = torch.cumsum(p, 0)
    mu = torch.cumsum(p * torch.arange(256, device=p.device), 0)
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w)).clamp_min(1e-12)
    return (int(torch.argmax(between[:-1])) + 1) / 256.0


def decide(x, nbg_all, rule, torch):
    """x: raw similarities [Q,H,W] with queries ordered [background, 26 bg words, fg classes 1..C-1].
    Returns (bg_mask [H,W] bool, keep [C-1] bool)."""
    nb = 1 if rule["V"] == 0 else nbg_all
    used = torch.cat([x[:nb], x[nbg_all:]], 0).float()
    p = (rule["T"] * used).softmax(0)
    pb = p[:nb].max(0)[0]
    pf = p[nb:]
    if rule["r"] > 0:
        k = max(1, int(0.01 * pf.shape[1] * pf.shape[2]))
        s = pf.flatten(1).topk(k, dim=1).values.mean(1)
        keep = s >= rule["r"] * s.max()
    else:
        keep = torch.ones(pf.shape[0], dtype=torch.bool, device=pf.device)
    pfk = pf.clone()
    pfk[~keep] = -1.0
    fgmax = pfk.max(0)[0]
    if rule["D"] == "argmax":
        bg = pb >= fgmax
    elif rule["D"] == "otsu":
        bg = fgmax < otsu_thr(fgmax.clamp_min(0), torch)
    else:
        bg = fgmax < float(rule["D"])
    return bg, keep


def main():
    method, mode, ds, split, out = sys.argv[1:6]
    smoke = int(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0
    rule = json.load(open(sys.argv[sys.argv.index("--rule") + 1])) if "--rule" in sys.argv else None
    global TAG
    if "--bgwords" in sys.argv:
        bgpath = sys.argv[sys.argv.index("--bgwords") + 1]
        bgw = json.load(open(bgpath))[ds]
        TAG = os.path.basename(bgpath).replace(".json", "")
    else:
        bgw = list(BG26)
    assert "background" not in bgw
    out = os.path.abspath(out)
    repo = f"{W2}/repos/{method}"
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.environ.setdefault("TORCH_HOME", f"{W2}/torch_home")

    import torch
    import torch.nn.functional as F
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmseg.evaluation import IoUMetric
    from mmseg.registry import METRICS, MODELS

    __import__(MODULES[method])
    if os.path.exists(f"{repo}/custom_datasets.py"):
        __import__("custom_datasets")

    @METRICS.register_module(force=True)
    class IoUMetricDump(IoUMetric):
        def __init__(self, dump_path=None, **kw):
            super().__init__(**kw)
            self.dump_path = dump_path

        def compute_metrics(self, results):
            cols = list(zip(*results))
            np.savez_compressed(self.dump_path, **{k: torch.stack(list(v)).cpu().numpy() for k, v in
                                                   zip(("intersect", "union", "pred", "label"), cols)})
            return super().compute_metrics(results)

    cfg = Config.fromfile(f"configs/cfg_{ds}.py")
    official_model = dict(cfg.model)
    model = dict(Config.fromfile("configs/base_config.py").model)
    cls = MODELS.get(official_model["type"])
    params = inspect.signature(cls.__init__).parameters
    npath, n_cls = name_file(method, ds, bgw)
    model["name_path"] = npath
    model["prob_thd"] = 0.0
    if "area_thd" in params:
        model["area_thd"] = None
    if "pamr_steps" in params:
        model["pamr_steps"] = 0
    if method == "ResCLIP":  # phase-2 deviations 6 / 6a
        model.update(delete_same_entity=True, attn_rcs_weights=[2.0, 0.5], attn_sfr_weights=[2.1, 0.7], temp_thd=0.18)
    if method == "ClearCLIP":
        model["model_type"] = "ClearCLIP"
        model["ignore_residual"] = True
    if method == "Trident":
        model["sam_refinement"] = True
        model["sam_ckpt"] = f"{W2}/weights/sam_vit_b_01ec64.pth"
    cfg.model = model

    dl = cfg.test_dataloader
    dl.dataset.data_root = DATA[(ds, split)]
    if ds == "voc21":
        assert dl.dataset.ann_file.endswith("val.txt")
        dl.dataset.ann_file = "ImageSets/Segmentation/" + ("train.txt" if split == "dev" else "val.txt")
    dl.num_workers = 4
    for t in dl.dataset.pipeline:
        if t["type"] == "Resize":
            t["scale"] = (2048, P_SHORT)
    if smoke:
        dl.dataset.indices = smoke
    dump = out.replace(".json", "_perimage.npz")
    cfg.test_evaluator = dict(type="IoUMetricDump", iou_metrics=["mIoU"], dump_path=dump)
    cfg.work_dir = os.path.dirname(out) + "/work"
    cfg.default_hooks.pop("visualization", None)

    nbg_all = 1 + len(bgw)
    orig = cls.postprocess_result
    cache_dir = out.replace(".json", "_cache")
    state = {"n": 0, "checked": False}

    def wrapped(self, seg_logits, data_samples, *a, **k):
        if not state["checked"]:
            qi = self.query_idx.tolist() if hasattr(self.query_idx, "tolist") else list(self.query_idx)
            assert len(qi) == nbg_all + n_cls - 1, (len(qi), nbg_all, n_cls)
            assert qi[:nbg_all] == [0] * nbg_all and qi[nbg_all:] == list(range(1, n_cls)), qi[:40]
            state["checked"] = True
        if mode == "cache":
            os.makedirs(cache_dir, exist_ok=True)
            for i in range(seg_logits.shape[0]):
                x = seg_logits[i].float()
                gt = data_samples[i].gt_sem_seg.data[0].to(x.device)
                if x.shape[-2:] != gt.shape[-2:]:
                    x = F.interpolate(x[None], size=gt.shape[-2:], mode="bilinear", align_corners=False)[0]
                h, w = max(1, gt.shape[0] // DS), max(1, gt.shape[1] // DS)
                xs = F.interpolate(x[None], size=(h, w), mode="bilinear", align_corners=False)[0]
                gs = F.interpolate(gt[None, None].float(), size=(h, w), mode="nearest")[0, 0]
                np.savez(f"{cache_dir}/{state['n']:05d}.npz", x=xs.half().cpu().numpy(), y=gs.to(torch.uint8).cpu().numpy())
                state["n"] += 1
            return orig(self, seg_logits, data_samples, *a, **k)
        new = seg_logits.clone()
        for i in range(seg_logits.shape[0]):
            bg, keep = decide(seg_logits[i], nbg_all, rule, torch)
            nb = 1 if rule["V"] == 0 else nbg_all
            new[i, :nbg_all] = -10.0
            new[i, :nb] = torch.where(bg, 10.0, -10.0).to(new.dtype)
            drop = torch.nonzero(~keep).flatten() + nbg_all
            if len(drop):
                new[i, drop] = -10.0
        # merge the background queries (all carry the same +/-10 after the override; for V0 query 0 carries it)
        small = torch.cat([new[:, :1], new[:, nbg_all:]], 1)
        qi_orig = self.query_idx
        if hasattr(qi_orig, "device"):
            self.query_idx = torch.arange(n_cls, device=qi_orig.device, dtype=qi_orig.dtype)
        else:
            self.query_idx = list(range(n_cls))
        try:
            return orig(self, small, data_samples, *a, **k)
        finally:
            self.query_idx = qi_orig

    cls.postprocess_result = wrapped
    t0 = time.time()
    res = Runner.from_cfg(cfg).test()
    pi = np.load(dump)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = pi["intersect"].sum(0) / pi["union"].sum(0)
    rec = {"method": method, "mode": mode, "dataset": ds, "split": split, "rule": rule, "mIoU": float(res["mIoU"]),
           "mIoU_recomputed": float(np.nanmean(iou) * 100), "iou": [None if np.isnan(v) else float(v) for v in iou],
           "n_images": int(pi["intersect"].shape[0]), "n_cached": state["n"], "seconds": time.time() - t0,
           "smoke": smoke, "name_path": npath, "n_bg_words": len(bgw), "gpu": torch.cuda.get_device_name(0)}
    json.dump(rec, open(out, "w"), indent=1)
    print("RESULT", method, mode, ds, split, round(rec["mIoU"], 2), "images", rec["n_images"], f"{rec['seconds']:.0f}s")


if __name__ == "__main__":
    main()
