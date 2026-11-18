"""S1: run one (method, setting, derived dataset) cell. Copy of V2-P2 run_one.py (md5 e1ced23a) with only these changes:
the dataset is an S1 derived variant described by /root/autodl-tmp/s1/specs/VARIANT.json (parent config, data root,
compact class list, P* and BGB query-name files); the dataset class is a subclass of the parent dataset class whose
METAINFO lists the derived classes. Settings: P (P*) | BGB. Everything else (P* rules, ResCLIP P* values, ClearCLIP
model_type, Trident SAM refinement, background-key transplant) is unchanged.
Usage: python s1_run_one.py METHOD SETTING VARIANT OUT_JSON [--smoke N]"""

import inspect
import json
import os
import sys
import time

import numpy as np

W = "/root/autodl-tmp/p2"
S1 = "/root/autodl-tmp/s1"
REPOS = {"SCLIP": "SCLIP", "ClearCLIP": "ClearCLIP", "NACLIP": "NACLIP", "ResCLIP": "ResCLIP", "SC-CLIP": "SC-CLIP",
         "ProxyCLIP": "ProxyCLIP", "Trident": "Trident"}
MODULES = {"SCLIP": ["clip_segmentor"], "ClearCLIP": ["clearclip_segmentor"], "NACLIP": ["naclip"],
           "ResCLIP": ["resclip_segmentor"], "SC-CLIP": ["scclip_segmentor"], "ProxyCLIP": ["proxyclip_segmentor"],
           "Trident": ["trident"]}
P_SHORT = 448
BG_KEYS = ("name_path", "prob_thd", "area_thd", "logit_scale")


def main():
    method, setting, variant, out = sys.argv[1:5]
    assert setting in ("P", "BGB")
    smoke = int(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0
    spec = json.load(open(f"{S1}/specs/{variant}.json"))
    out = os.path.abspath(out)
    repo = f"{W}/repos/{REPOS[method]}"
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.environ.setdefault("TORCH_HOME", f"{W}/torch_home")
    os.makedirs(os.path.expanduser("~/.cache/clip"), exist_ok=True)
    link = os.path.expanduser("~/.cache/clip/ViT-B-16.pt")
    if not os.path.exists(link):
        os.symlink(f"{W}/weights/ViT-B-16.pt", link)

    import torch
    import mmseg.datasets  # noqa: F401
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmseg.evaluation import IoUMetric
    from mmseg.registry import DATASETS, METRICS, MODELS

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

    cfg = Config.fromfile(f"configs/cfg_{spec['parent']}.py")
    official_model = dict(cfg.model)
    base_model = dict(Config.fromfile("configs/base_config.py").model)
    cls = MODELS.get(official_model["type"])
    params = inspect.signature(cls.__init__).parameters

    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(repo, p))

    model = dict(base_model)  # P*: base config only, unified names, no thresholds, no pixel post-processing
    model["name_path"] = spec["names_P"]
    model["prob_thd"] = 0.0
    if "area_thd" in params:
        model["area_thd"] = None
    if "pamr_steps" in params:
        model["pamr_steps"] = 0
    if method == "ResCLIP":  # V2-P2 CONTRACT dev. 6 / 6a
        model["delete_same_entity"] = True
        model["attn_rcs_weights"] = [2.0, 0.5]
        model["attn_sfr_weights"] = [2.1, 0.7]
        model["temp_thd"] = 0.18
    if setting == "BGB":  # transplant the parent dataset's official background trick
        for k in BG_KEYS:
            if k in official_model:
                model[k] = spec["names_BGB"][method] if k == "name_path" else official_model[k]
    if method == "ClearCLIP":
        model["model_type"] = "ClearCLIP"
        model["ignore_residual"] = True
    if method == "Trident":
        model["sam_refinement"] = True
        model["sam_ckpt"] = f"{W}/weights/sam_vit_b_01ec64.pth"
    cfg.model = model

    dl = cfg.test_dataloader
    parent_cls = DATASETS.get(dl.dataset.type)
    classes = tuple(spec["classes"])

    class S1Seg(parent_cls):
        METAINFO = dict(classes=classes, palette=[[(37 * i) % 256, (91 * i) % 256, (151 * i) % 256] for i in range(len(classes))])

    DATASETS.register_module(name="S1Seg", module=S1Seg, force=True)
    dl.dataset.type = "S1Seg"
    dl.dataset.data_root = spec["data_root"]
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

    t0 = time.time()
    runner = Runner.from_cfg(cfg)
    res = runner.test()
    pi = np.load(dump)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = pi["intersect"].sum(0) / pi["union"].sum(0)
    n_names = len([l for l in open(model["name_path"]).read().split("\n") if l.strip()])
    rec = {"method": method, "setting": setting, "dataset": variant, "parent": spec["parent"], "mIoU": float(res["mIoU"]),
           "mIoU_recomputed": float(np.nanmean(iou) * 100), "iou": [None if np.isnan(v) else float(v) for v in iou],
           "n_images": int(pi["intersect"].shape[0]), "n_classes": len(classes), "n_query_lines": n_names,
           "seconds": time.time() - t0, "smoke": smoke,
           "model_cfg": {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)) for k, v in model.items()},
           "short_side": [t["scale"] for t in dl.dataset.pipeline if t["type"] == "Resize"][0],
           "gpu": torch.cuda.get_device_name(0)}
    assert n_names == len(classes) == pi["intersect"].shape[1], (n_names, len(classes), pi["intersect"].shape)
    json.dump(rec, open(out, "w"), indent=1)
    print("RESULT", method, setting, variant, round(rec["mIoU"], 2), "images", rec["n_images"], f"{rec['seconds']:.0f}s")


if __name__ == "__main__":
    main()
