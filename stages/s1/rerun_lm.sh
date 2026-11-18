set -u
export PYTHONUTF8=1
W=/root/autodl-tmp/s1
cd $W/code
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
python -c "import sys; sys.argv=['x']; sys.path.insert(0,'.'); import s1_run_one_lm as lm; lm.patch('ProxyCLIP')" 2>&1 | grep S1_LM || { log "PATCH_PROXYCLIP_FAILED"; exit 2; }
for v in obj_bg30 obj_bg50; do for m in SCLIP NACLIP ResCLIP SC-CLIP ProxyCLIP; do
  f=$W/cells/"$m"__BGB__"$v".json; l=$W/cells/"$m"__BGB__"$v".log
  if [ ! -f $f ]; then
    [ -f $l ] && mv $l $W/cells/"$m"__BGB__"$v"_oom.log
    python s1_run_one_lm.py $m BGB $v $f > $l 2>&1; log "LM $m BGB $v rc=$? $(grep RESULT $l)"
  fi
done; done
log "RERUN DONE"
