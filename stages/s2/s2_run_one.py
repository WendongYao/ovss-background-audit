"""S2: run one cell. Same as S1 s1_run_one.py (P*, BGB unchanged) plus THR (P* + official numeric background keys,
P* name file) and NC (BGB with the S2 no-collision name file). All methods except Trident run through the S1
low-memory rewrite (s1_run_one_lm.patch; bitwise-identical, S1 deviation 6); Trident has no such line.
Usage: python s2_run_one.py METHOD SETTING VARIANT OUT_JSON [--smoke N]"""
import inspect
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/root/autodl-tmp/s1/code")
import s1_run_one_lm as lm  # noqa: E402
import s1_run_one as base  # noqa: E402

W, S1, S2 = base.W, base.S1, "/root/autodl-tmp/s2"
BG_KEYS = base.BG_KEYS


def main():
    method, setting, variant, out = sys.argv[1:5]
    assert setting in ("P", "BGB", "THR", "NC")
    smoke = int(sys.argv[sys.argv.index("--smoke") + 1]) if "--smoke" in sys.argv else 0
    spec = json.load(open(f"{S1}/specs/{variant}.json"))
    nc = json.load(open(f"{S2}/s2_collisions.json")).get(variant, {}).get(method, {}).get("nc_path")
    out = os.path.abspath(out)
    repo = f"{W}/repos/{base.REPOS[method]}"
    os.chdir(repo)
    sys.path.insert(0, repo)
    os.environ.setdefault("TORCH_HOME", f"{W}/torch_home")

    import torch
    import mmseg.datasets  # noqa: F401
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmseg.evaluation import IoUMetric
    from mmseg.registry import DATASETS, METRICS, MODELS

    for m in base.MODULES[method]:
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

    model = dict(base_model)
    model["name_path"] = spec["names_P"]
    model["prob_thd"] = 0.0
    if "area_thd" in params:
        model["area_thd"] = None
    if "pamr_steps" in params:
        model["pamr_steps"] = 0
    if method == "ResCLIP":
        model["delete_same_entity"] = True
        model["attn_rcs_weights"] = [2.0, 0.5]
        model["attn_sfr_weights"] = [2.1, 0.7]
        model["temp_thd"] = 0.18
    if setting in ("BGB", "THR", "NC"):
        for k in BG_KEYS:
            if k in official_model:
                if k == "name_path":
                    if setting == "BGB":
                        model[k] = spec["names_BGB"][method]
                    elif setting == "NC":
                        model[k] = nc
                    # THR keeps the P* name file
                else:
                    model[k] = official_model[k]
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
            t["scale"] = (2048, base.P_SHORT)
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
           "mIoU_recomputed": float(np.nanmean(iou) * 100), "iou": [None if np.isnan(x) else float(x) for x in iou],
           "n_images": int(pi["intersect"].shape[0]), "n_classes": len(classes), "n_query_lines": n_names,
           "name_path": model["name_path"], "seconds": time.time() - t0, "smoke": smoke,
           "model_cfg": {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v)) for k, v in model.items()},
           "gpu": torch.cuda.get_device_name(0)}
    assert n_names == len(classes) == pi["intersect"].shape[1], (n_names, len(classes), pi["intersect"].shape)
    json.dump(rec, open(out, "w"), indent=1)
    print("RESULT", method, setting, variant, round(rec["mIoU"], 2), "images", rec["n_images"], f"{rec['seconds']:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] != "Trident":
        lm.patch(sys.argv[1])
    main()
