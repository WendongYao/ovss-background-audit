#!/usr/bin/env bash
# S4 (planned on 416; see deviations for the instance actually used): render SCLIP predictions for the pre-selected images (selection.json sha256 75f6e05e...).
set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/s4
P2=${P2:-/root/autodl-tmp/p2/code/run_one.py}
S3=/root/autodl-tmp/s3/code/s3_run_one.py
VOC=758,391,63
COCO=2832,4844,1430
BED=3974,369,4575
# supplementary set (contract deviation 2; selection_supp.json sha256 451a3656...)
SVOC=14,190,224,1384,1201,1292,937,254,802,484
SCOCO=1651,2626,2812,1486,1082,730,4636,610,3110,2658
SBED=3209,2284,2168,3172,1142,365,3949,3502,599,4912
cd $W/code
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p $W/cells/{q1,q2,sq1,sq2} $W/out/{q1,q2,sq1,sq2}
run(){ # group script setting dataset idx
  local g=$1 sc=$2 s=$3 d=$4 i=$5
  local f=$W/cells/$g/SCLIP__${s}__${d}.json
  python s4_render.py $sc SCLIP $s $d $f $i $W/out/$g > $W/cells/$g/SCLIP__${s}__${d}.log 2>&1
  log "$g $s $d rc=$? $(grep -h 'RESULT\|S4_RENDERED' $W/cells/$g/SCLIP__${s}__${d}.log | tr '\n' ' ')"
}
for s in O P BGB; do run q1 $P2 $s voc21 $VOC; run q1 $P2 $s coco_object $COCO; done
run q1 $S3 THR voc21_orig $VOC
run q1 $S3 THR coco_object_orig $COCO
run q2 $P2 BGB coco_object $BED
run q2 $S3 NC coco_object_orig $BED
for s in O P BGB; do run sq1 $P2 $s voc21 $SVOC; run sq1 $P2 $s coco_object $SCOCO; done
run sq1 $S3 THR voc21_orig $SVOC
run sq1 $S3 THR coco_object_orig $SCOCO
run sq2 $P2 BGB coco_object $SBED
run sq2 $S3 NC coco_object_orig $SBED
cd $W && tar czf /root/autodl-tmp/s4_results.tar.gz cells out code && md5sum /root/autodl-tmp/s4_results.tar.gz
log "S4 DONE"
