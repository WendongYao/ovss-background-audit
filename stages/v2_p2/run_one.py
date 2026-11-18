"""Run one (method, setting, dataset) cell with the method's OFFICIAL repository code under mmseg.
Usage: python run_one.py METHOD SETTING DATASET OUT_JSON [--smoke N]
SETTING: O (official config) | P (fixed protocol P*) | BGB (P* + official background trick) | OTSU (P* + Otsu background rule)
Writes OUT_JSON with mIoU and per-class IoU, and OUT_JSON.replace('.json','_perimage.npz') with per-image
intersect/union/pred/label areas (for bootstrap)."""

import inspect
import json
import os
import sys
import time

import numpy as np

W = "/root/autodl-tmp/p2"
REPOS = {"SCLIP": "SCLIP", "ClearCLIP": "ClearCLIP", "NACLIP": "NACLIP", "ResCLIP": "ResCLIP", "SC-CLIP": "SC-CLIP",
         "ProxyCLIP": "ProxyCLIP", "Trident": "Trident"}
MODULES = {"SCLIP": ["clip_segmentor"], "ClearCLIP": ["clearclip_segmentor"], "NACLIP": ["naclip"],
           "ResCLIP": ["resclip_segmentor"], "SC-CLIP": ["scclip_segmentor"], "ProxyCLIP": ["proxyclip_segmentor"],
           "Trident": ["trident"]}
DATA = {"voc21": f"{W}/data/VOCdevkit/VOC2012", "voc20": f"{W}/data/VOCdevkit/VOC2012",
        "coco_object": f"{W}/data/coco_object", "ade20k": f"{W}/data/ade/ADEChallengeData2016"}
UNIFIED_NAMES = f"{W}/repos/ClearCLIP/configs"  # mmseg unified class names (as in ClearCLIP's repo)
P_SHORT = 448
BG_KEYS = ("name_path", "prob_thd", "area_thd", "logit_scale")


def main():
    method, setting, ds, out = sys.argv[1:5]
    smoke = int(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0
    repo = f"{W}/repos/{REPOS[method]}"
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.environ.setdefault("TORCH_HOME", f"{W}/torch_home")
    os.makedirs(os.path.expanduser("~/.cache/clip"), exist_ok=True)
    link = os.path.expanduser("~/.cache/clip/ViT-B-16.pt")
    if not os.path.exists(link):
        os.symlink(f"{W}/weights/ViT-B-16.pt", link)

    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmengine.structures import PixelData
    from mmseg.evaluation import IoUMetric
    from mmseg.registry import METRICS, MODELS

    for m in MODULES[method]:
        __import__(m)
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
    base_model = dict(Config.fromfile("configs/base_config.py").model)
    cls = MODELS.get(official_model["type"])
    params = inspect.signature(cls.__init__).parameters

    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(repo, p))

    if setting == "O":
        model = official_model
        model["name_path"] = resolve(model["name_path"])
    else:  # P*: base config only, unified names, no thresholds, no pixel post-processing
        model = dict(base_model)
        model["name_path"] = f"{UNIFIED_NAMES}/cls_{ds}.txt"
        model["prob_thd"] = 0.0
        if "area_thd" in params:
            model["area_thd"] = None
        if "pamr_steps" in params:
            model["pamr_steps"] = 0
        if method == "ResCLIP":  # base_config leaves required weights unset (constructor raises); rule in CONTRACT dev. 6
            model["delete_same_entity"] = True       # identical in all four official configs
            model["attn_rcs_weights"] = [2.0, 0.5]   # tau identical in all four configs; lambda_rcs = 0.5 stated in paper
            model["attn_sfr_weights"] = [2.1, 0.7]   # tau = median of official configs (2.2, 2.1, 1.8, 2.1); lambda_sfr = 0.7 in paper
            model["temp_thd"] = 0.18  # required too (None raises); varies, not in paper: median of 0.5, 0.1, 0.11, 0.25
        if setting == "BGB":  # transplant the official background trick
            for k in BG_KEYS:
                if k in official_model:
                    model[k] = resolve(official_model[k]) if k == "name_path" else official_model[k]
    if method == "ClearCLIP":  # the repo's base_config ships model_type='vanilla'; the published method is 'ClearCLIP'
        model["model_type"] = "ClearCLIP"
        model["ignore_residual"] = True
    if method == "Trident":  # main-table configuration uses SAM refinement (README: --sam_refine)
        model["sam_refinement"] = True
        model["sam_ckpt"] = f"{W}/weights/sam_vit_b_01ec64.pth"
    cfg.model = model

    dl = cfg.test_dataloader
    dl.dataset.data_root = DATA[ds]
    dl.num_workers = 4
    if setting != "O":
        for t in dl.dataset.pipeline:
            if t["type"] == "Resize":
                t["scale"] = (2048, P_SHORT)
    if smoke:
        dl.dataset.indices = smoke
    dump = out.replace(".json", "_perimage.npz")
    cfg.test_evaluator = dict(type="IoUMetricDump", iou_metrics=["mIoU"], dump_path=dump)
    cfg.work_dir = os.path.dirname(out) + "/work"
    cfg.default_hooks.pop("visualization", None)

    if setting == "OTSU":
        def otsu_post(self, seg_logits, data_samples, *a, **k):
            for i in range(seg_logits.shape[0]):
                probs = (seg_logits[i].float() * 100).softmax(0)
                num_cls = int(max(self.query_idx)) + 1
                if num_cls != len(self.query_idx):
                    ci = torch.nn.functional.one_hot(self.query_idx, num_cls).T.view(num_cls, len(self.query_idx), 1, 1)
                    probs = (probs.unsqueeze(0) * ci.to(probs.dtype)).max(1)[0]
                fg, arg = probs[1:].max(0)
                pred = arg + 1
                hist = torch.histc(fg, bins=256, min=0, max=1)
                p = hist / hist.sum()
                w = torch.cumsum(p, 0)
                mu = torch.cumsum(p * torch.arange(256, device=p.device), 0)
                between = (mu[-1] * w - mu) ** 2 / (w * (1 - w)).clamp_min(1e-12)
                thr = (int(torch.argmax(between[:-1])) + 1) / 256.0
                pred[fg < thr] = 0
                data_samples[i].set_data({"pred_sem_seg": PixelData(data=pred.unsqueeze(0))})
            return data_samples
        cls.postprocess_result = otsu_post

    t0 = time.time()
    runner = Runner.from_cfg(cfg)
    res = runner.test()
    pi = np.load(dump)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = pi["intersect"].sum(0) / pi["union"].sum(0)
    rec = {"method": method, "setting": setting, "dataset": ds, "mIoU": float(res["mIoU"]),
           "mIoU_recomputed": float(np.nanmean(iou) * 100), "iou": [None if np.isnan(v) else float(v) for v in iou],
           "n_images": int(pi["intersect"].shape[0]), "seconds": time.time() - t0, "smoke": smoke,
           "model_cfg": {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)) for k, v in model.items()},
           "short_side": [t["scale"] for t in dl.dataset.pipeline if t["type"] == "Resize"][0],
           "gpu": torch.cuda.get_device_name(0)}
    json.dump(rec, open(out, "w"), indent=1)
    print("RESULT", method, setting, ds, round(rec["mIoU"], 2), "images", rec["n_images"], f"{rec['seconds']:.0f}s")


if __name__ == "__main__":
    main()
