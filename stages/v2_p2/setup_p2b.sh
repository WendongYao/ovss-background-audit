#!/usr/bin/env bash
# V2-P2 setup on AutoDL (image: PyTorch 2.1.0 / Python 3.10 / CUDA 12.1). Safe to re-run: every step resumes.
# Works in no-GPU mode (无卡模式) except the final GPU self-test, which is skipped when no GPU is present.
# Usage:  bash setup_p2b.sh 2>&1 | tee -a /root/autodl-tmp/p2/setup_log.txt
set -u
W=/root/autodl-tmp/p2
mkdir -p $W/{repos,data/raw,weights,results} && cd $W
turbo(){ ( source /etc/network_turbo >/dev/null 2>&1; "$@" ); }  # AutoDL academic acceleration only for GitHub/HF/overseas weights
log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "1/6 python packages"
python -c "import torch,sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'py', sys.version.split()[0])"
pip install -q -U openmim
pip install -q "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install -q "requests>=2.31"  # openmim downgrades requests; JupyterLab needs >=2.31
pip install -q "mmengine==0.10.1" "mmsegmentation==1.2.2" ftfy regex einops "numpy<2" timm opencv-python-headless pillow scipy openpyxl
python -c "import mmcv, mmengine, mmseg; print('mmcv', mmcv.__version__, 'mmengine', mmengine.__version__, 'mmseg', mmseg.__version__)" || { log "FAILED: mm packages"; exit 1; }

log "2/6 official repositories (commit hashes recorded)"
cd $W/repos
for r in wangf3014/SCLIP mc-lan/ClearCLIP sinahmr/NACLIP yvhangyang/ResCLIP SuleBai/SC-CLIP mc-lan/ProxyCLIP YuHengsss/Trident facebookresearch/dino; do
  n=${r#*/}; [ -d $n/.git ] || turbo git clone -q https://github.com/$r.git $n || log "FAILED clone $r"
  echo "$r $(git -C $n rev-parse HEAD 2>/dev/null)" ; done | tee $W/results/repo_commits.txt
cd $W

log "3/6 weights"
cd $W/weights
get(){ turbo curl -sS -L -C - --retry 5 -o "$1" "$2" 2>&1 | grep -v "416\|resume" || true; }
get ViT-B-16.pt https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
get dino_vitbase8_pretrain.pth https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
get dino_vitbase16_pretrain.pth https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
get sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
sha256sum *.pt *.pth | tee $W/results/weights_sha256.txt
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
