#!/usr/bin/env bash
# S5 layer 3 on 416 (RTX 3090). Contract: experiments/CONTRACT_S5_CN.md
set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/s5
P4=/root/autodl-tmp/p4/code/p4_run.py
RULE=/root/autodl-tmp/p4/selected_rule.json
WORDS=/root/autodl-tmp/p4/bg_words_cc.json
CLSF=/root/autodl-tmp/p2/repos/ClearCLIP/configs/cls_voc21.txt
cd $W/code
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p $W/check $W/stage1 $W/stage2
log "smoke (SCLIP, dev, 20 images, LOO on)"
S5_LOO=1 python s5_loo.py $P4 SCLIP test voc21 dev $W/check/smoke.json --rule $RULE --bgwords $WORDS --smoke 20 > $W/check/smoke.log 2>&1
grep -h "RESULT\|S5_LOO" $W/check/smoke.log || { log "SMOKE FAILED"; tail -5 $W/check/smoke.log; exit 2; }
grep -q "replica_mismatch 0" $W/check/smoke.log || { log "SMOKE REPLICA MISMATCH - stop"; exit 3; }
log "implementation check (SCLIP CC voc21 val = 52.63 +-0.10, replica identical)"
S5_LOO=0 python s5_loo.py $P4 SCLIP test voc21 val $W/check/impl_SCLIP_val.json --rule $RULE --bgwords $WORDS > $W/check/impl.log 2>&1
grep -h "RESULT\|S5_LOO" $W/check/impl.log
python -c "
import json, sys
a = json.load(open('$W/check/impl_SCLIP_val.json'))['mIoU']; print('IMPL_CHECK', a); sys.exit(0 if abs(a - 52.63) <= 0.10 else 3)" || { log "IMPL_CHECK_FAILED - stop"; exit 3; }
grep -q "replica_mismatch 0" $W/check/impl.log || { log "IMPL REPLICA MISMATCH - stop"; exit 3; }
log "stage 1 (dev, LOO, 6 methods)"
for m in SCLIP ClearCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP; do
  f=$W/stage1/"$m"__voc21__dev
  [ -f $f.json ] || S5_LOO=1 python s5_loo.py $P4 $m test voc21 dev $f.json --rule $RULE --bgwords $WORDS > $f.log 2>&1
  log "$m rc=$? $(grep -h 'RESULT\|S5_LOO' $f.log | tr '\n' ' ')"
done
python s5_attr.py $W/stage1 $WORDS $CLSF $W/A.json $W/bg_words_cc_minusA.json > $W/attr.log 2>&1; cat $W/attr.log
python -c "import json,sys; sys.exit(0 if json.load(open('$W/A.json'))['A'] else 4)" || { log "A EMPTY - no stage 2 (contract)"; exit 0; }
log "stage 2 (val, CC minus A, 7 methods, unchanged p4_run.py)"
for m in SCLIP ClearCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP Trident; do
  f=$W/stage2/"$m"__voc21__CCminusA
  [ -f $f.json ] || python $P4 $m test voc21 val $f.json --rule $RULE --bgwords $W/bg_words_cc_minusA.json > $f.log 2>&1
  log "$m rc=$? $(grep -h RESULT $f.log)"
done
nvidia-smi --query-gpu=name --format=csv,noheader > $W/gpu.txt
cd /root/autodl-tmp && tar czf s5_results.tar.gz --exclude='s5/*/work' s5 && md5sum s5_results.tar.gz && stat -c %s s5_results.tar.gz
log "S5 DONE"
