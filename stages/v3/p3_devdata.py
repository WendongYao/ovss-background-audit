"""V3 dev data: 2,000 COCO train2017 images (seed 20261105) with COCO-Object labels rasterised from instances_train2017.json
(background 0, the 80 thing classes -> 1..80 in sorted category-id order, larger instances drawn first), laid out like the
validation COCO-Object folder. VOC dev uses VOC2012 train.txt directly (no files needed)."""

import io
import json
import os
import random
import zipfile

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

PUB = "/root/autodl-pub/COCO2017"
D = "/root/autodl-tmp/p2/data/coco_object_dev"
RAW = "/root/autodl-tmp/p2/data/raw"
os.makedirs(f"{D}/images/val2017", exist_ok=True)
os.makedirs(f"{D}/annotations/val2017", exist_ok=True)
ann = f"{RAW}/annotations/instances_train2017.json"
if not os.path.exists(ann):
    with zipfile.ZipFile(f"{PUB}/annotations_trainval2017.zip") as z:
        z.extract("annotations/instances_train2017.json", RAW)
coco = COCO(ann)
ids = sorted(coco.getImgIds())
sel = sorted(random.Random(20261105).sample(ids, 2000))
cats = sorted(coco.getCatIds())
assert len(cats) == 80
c2t = {c: i + 1 for i, c in enumerate(cats)}
with zipfile.ZipFile(f"{PUB}/train2017.zip") as z:
    names = set(z.namelist())
    for n, i in enumerate(sel):
        info = coco.loadImgs(i)[0]
        fn = info["file_name"]
        dst = f"{D}/images/val2017/{fn}"
        if not os.path.exists(dst):
            open(dst, "wb").write(z.read(f"train2017/{fn}"))
        lab = np.zeros((info["height"], info["width"]), np.uint8)
        anns = sorted(coco.loadAnns(coco.getAnnIds(imgIds=i, iscrowd=None)), key=lambda a: -a["area"])
        for a in anns:
            lab[coco.annToMask(a) > 0] = c2t[a["category_id"]]
        Image.fromarray(lab).save(f"{D}/annotations/val2017/{fn[:-4]}_instanceTrainIds.png")
json.dump({"seed": 20261105, "n": len(sel), "image_ids": sel}, open(f"{D}/dev_ids.json", "w"))
print("coco dev ok", len(os.listdir(f"{D}/images/val2017")), len(os.listdir(f"{D}/annotations/val2017")))
voc = open("/root/autodl-tmp/p2/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt").read().split()
val = set(open("/root/autodl-tmp/p2/data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt").read().split())
assert len(voc) == 1464 and not (set(voc) & val)
assert all(os.path.exists(f"/root/autodl-tmp/p2/data/VOCdevkit/VOC2012/SegmentationClass/{v}.png") for v in voc)
print("voc dev ok", len(voc))
