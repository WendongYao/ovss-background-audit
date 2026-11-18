#!/usr/bin/env bash
# S1 GPU stage: implementation check (ctx_bg0 = Context-60 through the S1 path) -> smoke -> Tier 2 -> Tier 3. Resumable.
set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/s1
cd $W/code
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p $W/cells $W/check
M7="SCLIP ClearCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP Trident"
cell(){ # method setting variant dir
  local m=$1 s=$2 v=$3 d=$4
  local f=$d/"$m"__"$s"__"$v".json
  local l=$d/"$m"__"$s"__"$v".log
  if [ ! -f $f ]; then python s1_run_one.py $m $s $v $f > $l 2>&1; log "$m $s $v rc=$? $(grep RESULT $l)"; fi
}
log "implementation check: SCLIP ctx_bg0 P 30.41, BGB 31.93 (+-0.05)"
cell SCLIP P ctx_bg0 $W/check
cell SCLIP BGB ctx_bg0 $W/check
python -c "
import json, sys
a = json.load(open('$W/check/SCLIP__P__ctx_bg0.json'))['mIoU']; b = json.load(open('$W/check/SCLIP__BGB__ctx_bg0.json'))['mIoU']
print('IMPL_CHECK', a, b); sys.exit(0 if abs(a - 30.41) <= 0.05 and abs(b - 31.93) <= 0.05 else 3)" || { log "IMPL_CHECK_FAILED - stop"; exit 3; }
log "smoke"
python s1_run_one.py SCLIP P ctx_bg70 $W/check/smoke_ctx.json --smoke 20 > $W/check/smoke_ctx.log 2>&1; grep RESULT $W/check/smoke_ctx.log || { log "SMOKE CTX FAILED"; tail -5 $W/check/smoke_ctx.log; exit 2; }
python s1_run_one.py Trident BGB obj_bg50 $W/check/smoke_obj.json --smoke 20 > $W/check/smoke_obj.log 2>&1; grep RESULT $W/check/smoke_obj.log || { log "SMOKE OBJ FAILED"; tail -5 $W/check/smoke_obj.log; exit 2; }
for v in ctx_bg30 ctx_bg50 ctx_bg70 obj_bg30 obj_bg50; do
  [ -f $W/specs/$v.json ] || { log "no spec $v, skip"; continue; }
  log "variant $v"
  for s in P BGB; do for m in $M7; do cell $m $s $v $W/cells; done; done
done
log "S1 DONE"
