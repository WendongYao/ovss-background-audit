"""V2-P2 data preparation (step 5 of setup). Same label conversion as phase 1.
COCO-Stuff val2017 label maps are the phase-1 files (transferred from the user's laptop as a tar, checked file by file
against an md5 manifest made on the laptop), because the official host served them at ~10 KB/s from AutoDL.
VOC and ADE20K validation labels are checked against md5 manifests of the phase-1 copies."""

import hashlib
import os
import tarfile
import zipfile

import numpy as np
from PIL import Image

W = '/root/autodl-tmp/p2/data'
R = f'{W}/raw'
T = '/root/autodl-tmp/p2/transfer'
OUT = '/root/autodl-tmp/p2/results/data_check.txt'


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def check(manifest, root):
    want = dict((l.split()[1], l.split()[0]) for l in open(manifest) if l.strip())
    bad = [k for k, v in want.items() if not os.path.exists(f'{root}/{k}') or md5(f'{root}/{k}') != v]
    return len(want), bad


log = []
# VOC
if not os.path.exists(f'{W}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'):
    os.makedirs(f'{W}/VOCdevkit', exist_ok=True)
    with tarfile.open(f'{R}/VOC2012.tar.gz') as t:  # AutoDL repack: top-level folder VOC2012/
        t.extractall(f'{W}/VOCdevkit', members=[m for m in t if m.name.startswith((
            'VOC2012/JPEGImages/', 'VOC2012/SegmentationClass/', 'VOC2012/ImageSets/Segmentation/'))])
ids = open(f'{W}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt').read().split()
assert len(ids) == 1449 and all(os.path.exists(f'{W}/VOCdevkit/VOC2012/JPEGImages/{i}.jpg') for i in ids)
n, bad = check(f'{T}/manifest_voc_val_labels.md5', f'{W}/VOCdevkit/VOC2012')
log.append(f'VOC val labels vs phase 1: {n} files, {len(bad)} differ {bad[:5]}')
# ADE20K validation
ade = f'{W}/ade/ADEChallengeData2016'
if len(os.listdir(f'{ade}/annotations/validation') if os.path.isdir(f'{ade}/annotations/validation') else []) < 2000:
    with zipfile.ZipFile(f'{R}/ADEChallengeData2016.zip') as z:
        z.extractall(f'{W}/ade', members=[n for n in z.namelist() if '/validation/' in n])
assert len(os.listdir(f'{ade}/images/validation')) == 2000
n, bad = check(f'{T}/manifest_ade_val_labels.md5', f'{ade}/annotations/validation')
log.append(f'ADE val labels vs phase 1: {n} files, {len(bad)} differ {bad[:5]}')
# COCO-Object: SCLIP/GroupViT cvt_coco_object.py mapping (things 0..90 -> 1..80, stuff and 255 -> 0)
co = f'{W}/coco_object'
os.makedirs(f'{co}/annotations/val2017', exist_ok=True)
if len(os.listdir(f'{co}/images/val2017') if os.path.isdir(f'{co}/images/val2017') else []) < 5000:
    with zipfile.ZipFile(f'{R}/val2017.zip') as z:
        z.extractall(f'{co}/images')
if len(os.listdir(f'{W}/stuffthingmaps/val2017') if os.path.isdir(f'{W}/stuffthingmaps/val2017') else []) < 5000:
    with tarfile.open(f'{T}/stuffthingmaps_val2017_phase1.tar') as t:
        t.extractall(f'{W}/stuffthingmaps')
n, bad = check(f'{T}/manifest_stuffthingmaps_val2017.md5', f'{W}/stuffthingmaps')
log.append(f'COCO-Stuff val maps vs phase 1: {n} files, {len(bad)} differ {bad[:5]}')
assert not bad
if len(os.listdir(f'{co}/annotations/val2017')) < 5000:
    thing = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 30, 31, 32, 33,
             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
             62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89]
    lut = np.zeros(256, np.uint8)
    for tr, k in enumerate(thing):
        lut[k] = tr + 1
    for f in sorted(os.listdir(f'{W}/stuffthingmaps/val2017')):
        a = np.array(Image.open(f'{W}/stuffthingmaps/val2017/{f}'))
        Image.fromarray(lut[a]).save(f'{co}/annotations/val2017/{f[:-4]}_instanceTrainIds.png')
assert len(os.listdir(f'{co}/images/val2017')) == 5000 and len(os.listdir(f'{co}/annotations/val2017')) == 5000
log.append('datasets ok: voc 1449, coco_object 5000, ade 2000')
open(OUT, 'w').write('\n'.join(log) + '\n')
print('\n'.join(log))
