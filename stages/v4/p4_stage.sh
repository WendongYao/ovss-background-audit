#!/usr/bin/env bash
# V4: model download -> vocabulary -> implementation check -> smoke -> 14 test cells. Resumable.
set -u
W=/root/autodl-tmp/p4
M=/root/autodl-tmp/models/Qwen2.5-7B-Instruct
cd $W/code
log(){ echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p $W/test $W/check /root/autodl-tmp/models
if [ ! -f $M/.done ]; then
  log "download Qwen2.5-7B-Instruct (hf-mirror)"
  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $M --local-dir-use-symlinks False > $W/download.log 2>&1 \
    || { log "hf-mirror failed, trying modelscope"; pip install -q modelscope > /dev/null 2>&1; modelscope download --model qwen/Qwen2.5-7B-Instruct --local_dir $M >> $W/download.log 2>&1; }
  ls $M/*.safetensors > /dev/null 2>&1 || { log "MODEL DOWNLOAD FAILED"; tail -5 $W/download.log; exit 2; }
  (cd $M && sha256sum *.safetensors config.json tokenizer.json > $W/model_sha256.txt) && touch $M/.done
fi
if [ ! -f $W/bg_words_cc.json ]; then
  log "generate vocabulary"
  python p4_gen_vocab.py $M $W > $W/gen_vocab.log 2>&1 || { log "VOCAB FAILED"; tail -5 $W/gen_vocab.log; exit 2; }
  sha256sum $W/bg_words_cc.json $W/cc_raw_generations.json | tee $W/vocab_sha256.txt
fi
grep -v Warning $W/gen_vocab.log | tail -3
log "implementation check (default 26 words: SCLIP 57.61, Trident 66.05 on VOC21 val, +-0.05)"
for m in SCLIP Trident; do
  [ -f $W/check/"$m"__voc21__SEL26.json ] || python p4_run.py $m test voc21 val $W/check/"$m"__voc21__SEL26.json --rule $W/selected_rule.json > $W/check/"$m"__voc21__SEL26.log 2>&1
done
python -c "
import json, sys
a = json.load(open('$W/check/SCLIP__voc21__SEL26.json'))['mIoU']; b = json.load(open('$W/check/Trident__voc21__SEL26.json'))['mIoU']
print('IMPL_CHECK', a, b); sys.exit(0 if abs(a - 57.61) <= 0.05 and abs(b - 66.05) <= 0.05 else 3)" || { log "IMPL_CHECK_FAILED - stop"; exit 3; }
log "smoke with CC vocabulary"
python p4_run.py SCLIP test voc21 val $W/check/SCLIP__voc21__CC_smoke.json --rule $W/selected_rule.json --bgwords $W/bg_words_cc.json --smoke 20 > $W/check/SCLIP__voc21__CC_smoke.log 2>&1
grep RESULT $W/check/SCLIP__voc21__CC_smoke.log || { log "SMOKE FAILED"; tail -5 $W/check/SCLIP__voc21__CC_smoke.log; exit 2; }
log "test cells"
for m in SCLIP ClearCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP Trident; do for d in voc21 coco_object; do
  if [ ! -f $W/test/"$m"__"$d"__CC.json ]; then
    python p4_run.py $m test $d val $W/test/"$m"__"$d"__CC.json --rule $W/selected_rule.json --bgwords $W/bg_words_cc.json > $W/test/"$m"__"$d"__CC.log 2>&1
    log "$m $d CC rc=$? $(grep RESULT $W/test/"$m"__"$d"__CC.log)"
  fi
done; done
log "V4 DONE"
