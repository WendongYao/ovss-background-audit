#!/usr/bin/env bash
# S4 deviation 6 on 416 (RTX 3090): (B) full ladder + intermediate maps for bed images 369, 4575;
# (R) re-run every THR / NC cell of S4 with the unchanged s4_render.py.
set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/s4
P2=/root/autodl-tmp/p2/code/run_one.py
S3=/root/autodl-tmp/s3/code/s3_run_one.py
B=369,4575
VOC=758,391,63; COCO=2832,4844,1430; BED=3974,369,4575
SVOC=14,190,224,1384,1201,1292,937,254,802,484
SCOCO=1651,2626,2812,1486,1082,730,4636,610,3110,2658
SBED=3209,2284,2168,3172,1142,365,3949,3502,599,4912
cd $W/code
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p $W/cells/b $W/out/b $W/cells/r3090/{q1,q2,sq1,sq2} $W/out/r3090/{q1,q2,sq1,sq2}
runb(){ # setting script dataset
  local f=$W/cells/b/SCLIP__$1__$3
  python s4b_render.py $2 SCLIP $1 $3 $f.json $B $W/out/b > $f.log 2>&1
  log "b $1 $3 rc=$? $(grep -h 'RESULT\|S4B_RENDERED' $f.log | tr '\n' ' ')"
}
runr(){ # group setting dataset idx
  local f=$W/cells/r3090/$1/SCLIP__$2__$3
  python s4_render.py $S3 SCLIP $2 $3 $f.json $4 $W/out/r3090/$1 > $f.log 2>&1
  log "r3090 $1 $2 $3 rc=$? $(grep -h 'RESULT\|S4_RENDERED' $f.log | tr '\n' ' ')"
}
runb P $P2 coco_object
runb BGB $P2 coco_object
runb O $P2 coco_object
runb THR $S3 coco_object_orig
runb NC $S3 coco_object_orig
runr q1 THR voc21_orig $VOC
runr q1 THR coco_object_orig $COCO
runr q2 NC coco_object_orig $BED
runr sq1 THR voc21_orig $SVOC
runr sq1 THR coco_object_orig $SCOCO
runr sq2 NC coco_object_orig $SBED
mkdir -p $W/out/b/img
python -c "
import json, shutil, os
d = json.load(open('$W/out/b/SCLIP__P__coco_object__paths.json'))
for im in d['images']:
    shutil.copy(im['img_path'], '$W/out/b/img/coco_object__%d__%s' % (im['index'], os.path.basename(im['img_path'])))
print('copied', len(d['images']))"
nvidia-smi --query-gpu=name --format=csv,noheader > $W/cells/gpu_416.txt
cd $W && tar czf /root/autodl-tmp/s4b_results.tar.gz cells/b cells/r3090 cells/gpu_416.txt out/b out/r3090 code && md5sum /root/autodl-tmp/s4b_results.tar.gz && stat -c %s /root/autodl-tmp/s4b_results.tar.gz
log "S4B DONE"
