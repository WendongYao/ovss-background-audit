#!/usr/bin/env bash
# V3 stage 1: dependencies, dev data, smoke tests (cache + test mode + selection on smoke caches).
set -u
W=/root/autodl-tmp/p3
cd $W/code
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
log "deps"
pip install -q pycocotools==2.0.7 2>&1 | grep -v WARNING
python -c "import pycocotools; print('pycocotools ok')"
log "dev data"
python p3_devdata.py || { log "FAILED devdata"; exit 1; }
log "smoke"
mkdir -p $W/smoke/cache_root
for m in SCLIP Trident; do for d in voc21 coco_object; do
  python p3_run.py $m cache $d dev $W/smoke/"$m"__"$d"__cache.json --smoke 20 > $W/smoke/"$m"__"$d"__cache.log 2>&1
  echo "$m $d cache rc=$? $(grep RESULT $W/smoke/"$m"__"$d"__cache.log)"
  ln -sfn $W/smoke/"$m"__"$d"__cache_cache $W/smoke/cache_root/"$m"__"$d"
done; done
echo '{"V": 0, "T": 100, "r": 0.0, "D": "otsu"}' > $W/rule_R0.json
python p3_run.py SCLIP test voc21 val $W/smoke/SCLIP__voc21__R0.json --rule $W/rule_R0.json --smoke 20 > $W/smoke/SCLIP__voc21__R0.log 2>&1
echo "SCLIP R0 smoke rc=$? $(grep RESULT $W/smoke/SCLIP__voc21__R0.log)"
python p3_run.py Trident test voc21 val $W/smoke/Trident__voc21__R0.json --rule $W/rule_R0.json --smoke 20 > $W/smoke/Trident__voc21__R0.log 2>&1
echo "Trident R0 smoke rc=$? $(grep RESULT $W/smoke/Trident__voc21__R0.log)"
python p3_select.py $W/smoke/cache_root $W/smoke/select_smoke.json > $W/smoke/select_smoke.log 2>&1
echo "select rc=$?"; tail -4 $W/smoke/select_smoke.log
log "STAGE1 DONE"
