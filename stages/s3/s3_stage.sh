#!/usr/bin/env bash
set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/s3
cd $W/code
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p $W/cells $W/check
M7="SCLIP ClearCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP Trident"
zero(){ python -c "import json,sys; sys.exit(0 if json.load(open('$W/s3_collisions.json'))['$2']['$1']['zero'] else 1)"; }
cell(){
  local m=$1 s=$2 v=$3 d=$4
  local f=$d/"$m"__"$s"__"$v".json
  local l=$d/"$m"__"$s"__"$v".log
  if [ "$s" = NC ] && zero $m $v; then log "$m NC $v zero collisions - not run (equals V2-P2 BGB)"; return; fi
  if [ ! -f $f ]; then python s3_run_one.py $m $s $v $f > $l 2>&1; log "$m $s $v rc=$? $(grep RESULT $l)"; fi
}
log "implementation check: ClearCLIP BGB voc21 51.80, SCLIP P voc21 35.91, ClearCLIP BGB coco_object 33.04 (+-0.05)"
cell ClearCLIP BGB voc21_orig $W/check
cell SCLIP P voc21_orig $W/check
cell ClearCLIP BGB coco_object_orig $W/check
python -c "
import json, sys
a = json.load(open('$W/check/ClearCLIP__BGB__voc21_orig.json'))['mIoU']; b = json.load(open('$W/check/SCLIP__P__voc21_orig.json'))['mIoU']; c = json.load(open('$W/check/ClearCLIP__BGB__coco_object_orig.json'))['mIoU']
print('IMPL_CHECK', a, b, c); sys.exit(0 if abs(a - 51.80) <= 0.05 and abs(b - 35.91) <= 0.05 and abs(c - 33.04) <= 0.05 else 3)" || { log "IMPL_CHECK_FAILED - stop"; exit 3; }
log "smoke"
python s3_run_one.py SCLIP THR voc21_orig $W/check/smoke_thr.json --smoke 20 > $W/check/smoke_thr.log 2>&1; grep RESULT $W/check/smoke_thr.log || { log "SMOKE THR FAILED"; tail -5 $W/check/smoke_thr.log; exit 2; }
python s3_run_one.py ProxyCLIP NC coco_object_orig $W/check/smoke_nc.json --smoke 20 > $W/check/smoke_nc.log 2>&1; grep RESULT $W/check/smoke_nc.log || { log "SMOKE NC FAILED"; tail -5 $W/check/smoke_nc.log; exit 2; }
for v in voc21_orig coco_object_orig; do
  log "variant $v"
  for s in THR NC; do for m in $M7; do cell $m $s $v $W/cells; done; done
done
log "S3 DONE"
