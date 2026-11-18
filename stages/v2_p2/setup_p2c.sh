#!/usr/bin/env bash
# V2-P2 setup, part C: replaces steps 3-6 of setup_p2b.sh (which was stopped during the weights step because the
# accelerated route was slower than the direct one: direct ~0.39 MB/s vs accelerated ~0.12 MB/s). Weights download
# in parallel over the direct route (resuming partial files) while the datasets are prepared. Same sources, same checks.
set -u
W=/root/autodl-tmp/p2
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
cd $W/weights
log "3/6 weights (parallel, direct route)"
dl(){ for i in 1 2 3 4 5 6; do curl -sS -L -C - --retry 5 -o "$1" "$2" 2>/dev/null; s=$?; [ $s -eq 0 -o $s -eq 33 ] && break; sleep 5; done; echo "  $1 done (curl exit $s)"; }
dl dino_vitbase8_pretrain.pth https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth &
dl dino_vitbase16_pretrain.pth https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth &
dl sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth &
cd $W
log "4/6 datasets: VOC2012, COCO val2017 and ADE20K from AutoDL public data (/root/autodl-pub); COCO-Stuff maps downloaded"
cd $W/data/raw
P=/root/autodl-pub
[ -s VOC2012.tar.gz ] || cp $P/VOCdevkit/VOC2012.tar.gz .
ln -sf $P/COCO2017/val2017.zip val2017.zip
ln -sf $P/ADEChallengeData2016/ADEChallengeData2016.zip ADEChallengeData2016.zip
md5sum val2017.zip | tee $W/results/data_md5.txt   # phase-1 copy: 442b8da7639aecaf257c1dceb8ba8c80
ls -laL VOC2012.tar.gz val2017.zip ADEChallengeData2016.zip | tee -a $W/results/data_md5.txt   # phase-1 ADE zip size 967382037
for i in 1 2 3 4 5; do curl -sS -L -C - --retry 3 -o stuffthingmaps_trainval2017.zip https://calvin-vision.net/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip && break; sleep 10; done
ls -la stuffthingmaps_trainval2017.zip   # expected 659008564
cd $W

log "5/6 extract and convert (same label conversion as phase 1)"
python - <<'EOF'
import os, tarfile, zipfile, numpy as np
from PIL import Image
W = '/root/autodl-tmp/p2/data'
R = f'{W}/raw'
# VOC
if not os.path.exists(f'{W}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'):
    os.makedirs(f'{W}/VOCdevkit', exist_ok=True)
    with tarfile.open(f'{R}/VOC2012.tar.gz') as t:  # AutoDL repack: top-level folder VOC2012/
        t.extractall(f'{W}/VOCdevkit', members=[m for m in t if m.name.startswith(('VOC2012/JPEGImages/',
                     'VOC2012/SegmentationClass/', 'VOC2012/ImageSets/Segmentation/'))])
ids = open(f'{W}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt').read().split()
assert len(ids) == 1449 and all(os.path.exists(f'{W}/VOCdevkit/VOC2012/JPEGImages/{i}.jpg') for i in ids)
# ADE20K validation
ade = f'{W}/ade/ADEChallengeData2016'
if len(os.listdir(f'{ade}/annotations/validation') if os.path.isdir(f'{ade}/annotations/validation') else []) < 2000:
    with zipfile.ZipFile(f'{R}/ADEChallengeData2016.zip') as z:
        z.extractall(f'{W}/ade', members=[n for n in z.namelist() if '/validation/' in n])
assert len(os.listdir(f'{ade}/images/validation')) == 2000
# COCO-Object: SCLIP/GroupViT cvt_coco_object.py mapping (things 0..90 -> 1..80, stuff and 255 -> 0)
co = f'{W}/coco_object'
os.makedirs(f'{co}/annotations/val2017', exist_ok=True)
if len(os.listdir(f'{co}/annotations/val2017')) < 5000:
    with zipfile.ZipFile(f'{R}/val2017.zip') as z:
        z.extractall(f'{co}/images')
    with zipfile.ZipFile(f'{R}/stuffthingmaps_trainval2017.zip') as z:
        z.extractall(f'{W}/stuffthingmaps', members=[n for n in z.namelist() if n.startswith('val2017/')])
    thing = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,30,31,32,33,34,35,36,37,38,39,40,41,
             42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,66,69,71,72,73,74,75,76,77,78,79,80,81,
             83,84,85,86,87,88,89]
    lut = np.zeros(256, np.uint8)
    for tr, k in enumerate(thing):
        lut[k] = tr + 1
    for f in sorted(os.listdir(f'{W}/stuffthingmaps/val2017')):
        a = np.array(Image.open(f'{W}/stuffthingmaps/val2017/{f}'))
        Image.fromarray(lut[a]).save(f'{co}/annotations/val2017/{f[:-4]}_instanceTrainIds.png')
assert len(os.listdir(f'{co}/images/val2017')) == 5000 and len(os.listdir(f'{co}/annotations/val2017')) == 5000
print('datasets ok: voc 1449, coco_object 5000, ade 2000')
EOF
[ $? -eq 0 ] || { log "FAILED: datasets"; exit 1; }

log "waiting for weights"; wait; cd $W/weights; sha256sum *.pt *.pth | tee $W/results/weights_sha256.txt; ls -la; cd $W
log "6/6 GPU self-test (skipped in no-GPU mode)"
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee $W/results/gpu.txt
  python - <<'EOF'
import torch, mmcv
from mmcv.ops import nms
x = torch.randn(64, 768, device='cuda', dtype=torch.float16)
print('matmul ok', float((x @ x.T).float().mean()))
b = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device='cuda')
print('mmcv cuda op ok', nms(b, torch.tensor([0.9, 0.8], device='cuda'), 0.5)[1].tolist())
EOF
else
  echo "no GPU visible: self-test skipped (run this script again after switching to GPU mode)"
fi
log "SETUP DONE"
