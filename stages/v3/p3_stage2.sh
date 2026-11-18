#!/usr/bin/env bash
# V3 stage 2: dev caches -> rule selection -> implementation check -> test cells (single stream). Resumable.
set -u
W=/root/autodl-tmp/p3
cd $W/code
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p $W/dev $W/cache_root $W/test
log "dev caches"
for m in SCLIP SC-CLIP Trident; do for d in voc21 coco_object; do
  if [ ! -f $W/dev/"$m"__"$d"__cache.json ]; then
    python p3_run.py $m cache $d dev $W/dev/"$m"__"$d"__cache.json > $W/dev/"$m"__"$d"__cache.log 2>&1
    log "$m $d cache rc=$? $(grep RESULT $W/dev/"$m"__"$d"__cache.log)"
  fi
  ln -sfn $W/dev/"$m"__"$d"__cache_cache $W/cache_root/"$m"__"$d"
done; done
for m in SCLIP SC-CLIP Trident; do for d in voc21 coco_object; do
  [ -f $W/dev/"$m"__"$d"__cache.json ] || { log "MISSING CACHE $m $d"; exit 2; }
done; done
log "R0 full-resolution on SCLIP VOC dev (descriptive)"
[ -f $W/dev/SCLIP__voc21__R0_fullres.json ] || python p3_run.py SCLIP test voc21 dev $W/dev/SCLIP__voc21__R0_fullres.json --rule $W/rule_R0.json > $W/dev/SCLIP__voc21__R0_fullres.log 2>&1
grep RESULT $W/dev/SCLIP__voc21__R0_fullres.log
log "selection"
if [ ! -f $W/selected_rule.json ]; then
  python p3_select.py $W/cache_root $W/selected.json > $W/select.log 2>&1 || { log "SELECT FAILED"; tail -5 $W/select.log; exit 2; }
fi
tail -14 $W/select.log
sha256sum $W/selected.json $W/selected_rule.json | tee $W/selected_sha256.txt
log "implementation check (SCLIP R0 VOC21 val must be 49.79 +- 0.05)"
[ -f $W/test/SCLIP__voc21__R0.json ] || python p3_run.py SCLIP test voc21 val $W/test/SCLIP__voc21__R0.json --rule $W/rule_R0.json > $W/test/SCLIP__voc21__R0.log 2>&1
python -c "import json,sys; v=json.load(open('$W/test/SCLIP__voc21__R0.json'))['mIoU']; print('IMPL_CHECK', v); sys.exit(0 if abs(v-49.79)<=0.05 else 3)" || { log "IMPL_CHECK_FAILED - stop"; exit 3; }
log "test cells"
for m in SCLIP ClearCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP Trident; do for d in voc21 coco_object; do
  if [ ! -f $W/test/"$m"__"$d"__SEL.json ]; then
    python p3_run.py $m test $d val $W/test/"$m"__"$d"__SEL.json --rule $W/selected_rule.json > $W/test/"$m"__"$d"__SEL.log 2>&1
    log "$m $d SEL rc=$? $(grep RESULT $W/test/"$m"__"$d"__SEL.log)"
  fi
done; done
log "Trident R0 with SAM refinement (descriptive correction of phase 2)"
for d in voc21 coco_object; do
  if [ ! -f $W/test/Trident__"$d"__R0.json ]; then
    python p3_run.py Trident test $d val $W/test/Trident__"$d"__R0.json --rule $W/rule_R0.json > $W/test/Trident__"$d"__R0.log 2>&1
    log "Trident $d R0 rc=$? $(grep RESULT $W/test/Trident__"$d"__R0.log)"
  fi
done
log "STAGE2 DONE"
