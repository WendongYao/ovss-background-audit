#!/usr/bin/env bash
# S2 GPU stage: implementation check -> smoke -> layer A (obj_bg30) -> B (obj_bg50) -> C (ctx_bg70). Resumable.
set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/s2
cd $W/code
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p $W/cells $W/check
M7="SCLIP ClearCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP Trident"
zero(){ python -c "import json,sys; sys.exit(0 if json.load(open('$W/s2_collisions.json'))['$2']['$1']['zero'] else 1)"; }
cell(){ # method setting variant dir
  local m=$1 s=$2 v=$3 d=$4
  local f=$d/"$m"__"$s"__"$v".json
  local l=$d/"$m"__"$s"__"$v".log
  if [ "$s" = NC ] && zero $m $v; then log "$m NC $v zero collisions - not run (equals S1 BGB)"; return; fi
  if [ ! -f $f ]; then python s2_run_one.py $m $s $v $f > $l 2>&1; log "$m $s $v rc=$? $(grep RESULT $l)"; fi
}
log "implementation check: ClearCLIP BGB obj_bg30 18.12, SCLIP P ctx_bg70 20.08 (+-0.05)"
cell ClearCLIP BGB obj_bg30 $W/check
cell SCLIP P ctx_bg70 $W/check
python -c "
import json, sys
a = json.load(open('$W/check/ClearCLIP__BGB__obj_bg30.json'))['mIoU']; b = json.load(open('$W/check/SCLIP__P__ctx_bg70.json'))['mIoU']
print('IMPL_CHECK', a, b); sys.exit(0 if abs(a - 18.12) <= 0.05 and abs(b - 20.08) <= 0.05 else 3)" || { log "IMPL_CHECK_FAILED - stop"; exit 3; }
log "smoke"
python s2_run_one.py SCLIP THR obj_bg30 $W/check/smoke_thr.json --smoke 20 > $W/check/smoke_thr.log 2>&1; grep RESULT $W/check/smoke_thr.log || { log "SMOKE THR FAILED"; tail -5 $W/check/smoke_thr.log; exit 2; }
python s2_run_one.py ProxyCLIP NC obj_bg30 $W/check/smoke_nc.json --smoke 20 > $W/check/smoke_nc.log 2>&1; grep RESULT $W/check/smoke_nc.log || { log "SMOKE NC FAILED"; tail -5 $W/check/smoke_nc.log; exit 2; }
for v in obj_bg30 obj_bg50 ctx_bg70; do
  log "variant $v"
  for s in THR NC; do for m in $M7; do cell $m $s $v $W/cells; done; done
done
log "S2 DONE"
