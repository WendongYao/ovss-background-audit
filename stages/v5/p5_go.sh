#!/usr/bin/env bash
# V5 launcher (GPU mode): Context-60 conversion (if not yet done) then p5_stage.sh.
set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/p5
cd $W/code
if [ ! -f /root/autodl-tmp/p2/data/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt ]; then
  echo "[$(date '+%H:%M:%S')] context60 conversion"
  python p5_ctxdata.py > $W/ctxdata.log 2>&1 || { echo "CTXDATA FAILED"; tail -5 $W/ctxdata.log; exit 2; }
  tail -3 $W/ctxdata.log
fi
bash p5_stage.sh
