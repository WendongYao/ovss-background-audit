#!/usr/bin/env bash
# V5 (GPU mode): CF vocabulary -> implementation check -> Context-60 O (reproduction gate) -> remaining cells. Resumable.
set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/p5
M=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
cd $W/code
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p $W/cells $W/check
[ -f /root/autodl-tmp/p2/data/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt ] || { log "NO CONTEXT DATA"; exit 2; }
if [ ! -f $W/bg_words_cf.json ]; then
  log "CF vocabulary"
  python p5_vocab.py $M /root/autodl-tmp/p4/bg_words_cc.json $W > $W/vocab.log 2>&1 || { log "VOCAB FAILED"; tail -5 $W/vocab.log; exit 2; }
  sha256sum $W/bg_words_cf.json $W/bg_words_cc3.json $W/cf_judgements.json $W/cc_raw_generations_context60.json | tee $W/vocab_sha256.txt
fi
grep -v Warning $W/vocab.log | grep "CC\|CF" | cut -c1-300
log "implementation check (SCLIP VOC21: O 59.63, SEL26 57.60; +-0.05)"
[ -f $W/check/SCLIP__O__voc21.json ] || python p5_run_one.py SCLIP O voc21 $W/check/SCLIP__O__voc21.json > $W/check/SCLIP__O__voc21.log 2>&1
[ -f $W/check/SCLIP__voc21__SEL26.json ] || python p5_run.py SCLIP test voc21 val $W/check/SCLIP__voc21__SEL26.json --rule $W/selected_rule.json > $W/check/SCLIP__voc21__SEL26.log 2>&1
python -c "
import json, sys
a = json.load(open('$W/check/SCLIP__O__voc21.json'))['mIoU']; b = json.load(open('$W/check/SCLIP__voc21__SEL26.json'))['mIoU']
print('IMPL_CHECK', a, b); sys.exit(0 if abs(a - 59.63) <= 0.05 and abs(b - 57.60) <= 0.05 else 3)" || { log "IMPL_CHECK_FAILED - stop"; exit 3; }
M7="SCLIP ClearCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP Trident"
one(){ # setting method -> cell via p5_run_one (O/P/BGB on context60)
  local s=$1 m=$2
  local f=$W/cells/"$m"__"$s"__context60.json l=$W/cells/"$m"__"$s"__context60.log
  if [ ! -f $f ]; then python p5_run_one.py $m $s context60 $f > $l 2>&1; log "$m $s context60 rc=$? $(grep RESULT $l)"; fi
}
rule(){ # tag method dataset [bgwords]
  local tag=$1 m=$2 d=$3 bw=""
  local f=$W/cells/"$m"__"$tag"__"$d".json l=$W/cells/"$m"__"$tag"__"$d".log
  [ $# -ge 4 ] && bw=$4
  if [ ! -f $f ]; then
    if [ -n "$bw" ]; then python p5_run.py $m test $d val $f --rule $W/selected_rule.json --bgwords $bw > $l 2>&1
    else python p5_run.py $m test $d val $f --rule $W/selected_rule.json > $l 2>&1; fi
    log "$m $tag $d rc=$? $(grep RESULT $l)"
  fi
}
log "smoke"
python p5_run_one.py SCLIP O context60 $W/check/SCLIP__O__context60_smoke.json --smoke 20 > $W/check/smoke1.log 2>&1; grep RESULT $W/check/smoke1.log || { log "SMOKE O FAILED"; tail -5 $W/check/smoke1.log; exit 2; }
python p5_run.py SCLIP test context60 val $W/check/SCLIP__CF__context60_smoke.json --rule $W/selected_rule.json --bgwords $W/bg_words_cf.json --smoke 20 > $W/check/smoke2.log 2>&1; grep RESULT $W/check/smoke2.log || { log "SMOKE CF FAILED"; tail -5 $W/check/smoke2.log; exit 2; }
log "Context-60 official configs (reproduction gate)"
for m in $M7; do one O $m; done
python - <<EOF || { log "CHECK_DATA - reproduction gate failed, stop"; exit 4; }
import json, sys
T = {"SCLIP": 31.74, "ClearCLIP": 32.6, "NACLIP": 34.99, "ResCLIP": 33.5, "SC-CLIP": 36.8, "ProxyCLIP": 35.3, "Trident": 38.6}
ok = 0
for m, t in T.items():
    v = json.load(open(f"$W/cells/{m}__O__context60.json"))["mIoU"]; ok += abs(v - t) <= 1.0; print("G0", m, v, t, round(v - t, 2))
print("G0 within 1.0:", ok, "of 7"); sys.exit(0 if ok >= 4 else 4)
EOF
log "Context-60 P*, BGB, SEL26, CF"
for m in $M7; do one P $m; done
for m in $M7; do one BGB $m; done
for m in $M7; do rule SEL26 $m context60; done
for m in $M7; do rule CF $m context60 $W/bg_words_cf.json; done
log "CF on VOC21 and COCO-Object"
for m in $M7; do for d in voc21 coco_object; do rule CF $m $d $W/bg_words_cf.json; done; done
log "V5 DONE"
