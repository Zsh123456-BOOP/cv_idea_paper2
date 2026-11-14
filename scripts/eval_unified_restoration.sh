#!/usr/bin/env bash
# Frozen checkpoints, shared inputs, one OCR engine for every model: the validation split (the old test set is retired).
# Cards: the restorations go to every card with room at once (_common.sh open_card); GPU_A and GPU_B run the OCR (TP2), default cards 1
# and 2.  MODELS=<list> evaluates another list
# (candidate versions for eval.unified_restoration choose) in place of configs/unified_restoration.json.  DEV_EXPORT=<export of the
# test page> DEV_PAGES=<its page images> adds the development half of the test questions (configs/test_halves.json).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
OUT="$1"
require_clean_source
test ! -e "$OUT"
mkdir "$OUT"
trap 'touch "$OUT/FAILED"' ERR
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
O=/home/zsh/hme/outputs
LAB=$O/paper2_annotation_labels
"$PY" -m revrender.eval.unified_restoration prepare \
  --models "${MODELS:-$ROOT/configs/unified_restoration.json}" \
  --selection "$O/paper2_annotation_labels_allowed20261021/selection.json" \
  --labels "$LAB/涂改标注_paper2-revision-boxes-val_2026-10-05-08-08.json" \
  --human-validation "$O/paper2_fixed_validation20261007/validation_native.json" \
  --pages "$O/paper2_pages_clean_unlabeled3976" \
  ${DEV_EXPORT:+--dev-export "$DEV_EXPORT" --dev-pages "$DEV_PAGES" --dev-questions "$ROOT/configs/test_halves.json"} \
  --output-dir "$OUT/inputs"
mkdir "$OUT/runs"
export GPU_A=${GPU_A:-GPU-d0bcc564-5bd9-b943-f995-8536415088e3} GPU_B=${GPU_B:-GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0}
case "$GPU_A,$GPU_B" in *314fdf43*) export ALLOW_GPU3=1 ;; esac   # card 3 only when the caller found nobody on it
MODELS=${MODELS:-$ROOT/configs/unified_restoration.json}
# each model with the architecture its runtime follows (a candidate version's name carries its step)
"$PY" -c 'import json,sys
for m in json.load(open(sys.argv[1])):
    print(m["name"], m["type"] if m["type"] == "own" else json.load(open(m["run"]+"/run_manifest.json"))["config"]["model"])' "$MODELS" > "$OUT/kinds.txt"
restore_version() {   # CARD NUMA NAME KIND: one version's restorations of the inputs
  local card="$1" numa="$2" name="$3" kind="$4"
  RUNTIME="$PY"
  export PYTHONPATH="$ROOT"
  unset LD_LIBRARY_PATH
  if [ "$kind" = DE-GAN ]; then
    RUNTIME=/home/zsh/hme/envs/paper2_restoration_tf/bin/python
    export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
    for directory in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do
      export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
  elif [ "$kind" = StainRestorer ]; then
    export PYTHONPATH="$ROOT:/home/zsh/hme/envs/paper2_stain_deps"
  fi
  bash "$ROOT/scripts/launch_gpu_job.sh" "$card" "$numa" "$OUT/runs/$name" -- \
    "$RUNTIME" -m revrender.eval.unified_restoration infer --inputs "$OUT/inputs" --name "$name" \
    --output-dir "$OUT/runs/$name" > "$OUT/runs/$name.log" 2>&1 < /dev/null
}
# every version on whichever card has room (open_card: two of our jobs a card at most, card 3 only while nobody else is on it)
while read -r name kind; do place_job 14000 restore_version "$name" "$kind"; done < "$OUT/kinds.txt"
wait
while read -r name kind; do test -e "$OUT/runs/$name/run_manifest.json" || { echo "no restorations: $name"; exit 1; }; done < "$OUT/kinds.txt"
unset LD_LIBRARY_PATH
export PYTHONPATH="$ROOT"
"$PY" -m revrender.eval.unified_restoration merge --inputs "$OUT/inputs" --runs "$OUT/runs" --output-dir "$OUT/pack"
bash "$ROOT/scripts/finish_unified_restoration.sh" "$OUT"
