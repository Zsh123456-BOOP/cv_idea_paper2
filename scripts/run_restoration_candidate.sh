#!/usr/bin/env bash
# Released-weight validation followed by independent, from-scratch task adaptation.
# Usage: CARD MODEL DATA UPSTREAM WEIGHTS VALIDATION_INPUT OUT
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
CARD="$1"; MODEL="$2"; DATA="$3"; UPSTREAM="$4"; WEIGHTS="$5"; INPUT="$6"; OUT="$7"
case "$CARD" in
  0) GPU=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; NUMA=0 ;;
  2) GPU=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; NUMA=1 ;;
  3) GPU=GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b; NUMA=1 ;;
  *) echo 'only cards 0/2/3 authorized'; exit 2 ;;
esac
require_clean_source
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
mkdir "$OUT"
trap 'code=$?; if [ "$code" -ne 0 ]; then touch "$OUT/FAILED"; fi' EXIT
export ALLOW_GPU3=1 PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
IDLE=0; [ "$CARD" != 3 ] || IDLE=1
HUMAN=/home/zsh/hme/outputs/paper2_fixed_validation20261007/validation_native.json

# Test training and checkpoint reload before committing to a long run.
bash "$ROOT/scripts/run_restoration_suite.sh" "$CARD" "$DATA" "$UPSTREAM" "$OUT/smoke" smoke "$CARD" "$MODEL"

"$PY" -m revrender.baselines.pretrained catalogue --model "$MODEL" --weights "$WEIGHTS" --output-dir "$OUT/catalogue"
"$PY" -c 'import json,sys; print("\n".join(r["name"] for r in json.load(open(sys.argv[1]))))' \
  "$OUT/catalogue/candidates.json" > "$OUT/candidates.txt"
mkdir "$OUT/released"
RUNS=()
while IFS= read -r candidate; do
  dest="$OUT/released/$candidate"
  wait_for_gpu_memory "$GPU" 18000 "$IDLE"
  bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" "$NUMA" "$dest" -- \
    "$PY" -m revrender.baselines.pretrained infer --candidate "$candidate" \
    --catalogue "$OUT/catalogue/candidates.json" --weights "$WEIGHTS" --upstream "$UPSTREAM" \
    --human-validation "$HUMAN" --input-pack "$INPUT" --output-dir "$dest" > "$dest.log" 2>&1
  RUNS+=("$dest")
done < "$OUT/candidates.txt"
"$PY" -m revrender.baselines.restoration_export merge --inputs "${RUNS[@]}" --output-dir "$OUT/released/pack"
OCR_MEMORY=.9; OCR_FREE=22500
if [ "$CARD" = 2 ]; then OCR_MEMORY=.85; OCR_FREE=21000; fi
wait_for_gpu_memory "$GPU" "$OCR_FREE" "$IDLE"
bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" "$NUMA" "$OUT/released/qwen8b" -- env CUDA_VISIBLE_DEVICES="$CARD" \
  "$PY" -m revrender.recognize_lines --pack "$OUT/released/pack" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct \
  --conditions erased_neutral --gpu-memory "$OCR_MEMORY" --output-dir "$OUT/released/qwen8b" > "$OUT/released/qwen8b.log" 2>&1
"$PY" -m revrender.eval.revision_boxes read --pack "$OUT/released/pack" \
  --readings "$OUT/released/qwen8b/predictions.jsonl" --output "$OUT/released/read_qwen8b.json"
"$PY" -m revrender.baselines.pretrained report --pack "$OUT/released/pack" \
  --readings "$OUT/released/qwen8b/predictions.jsonl" --scores "$OUT/released/read_qwen8b.json" \
  --runs "${RUNS[@]}" --output-dir "$OUT/released/report"
touch "$OUT/released/COMPLETED"

# Selection above is diagnostic only: formal training never loads those weights.
bash "$ROOT/scripts/run_restoration_suite.sh" "$CARD" "$DATA" "$UPSTREAM" "$OUT/formal" formal "$CARD" "$MODEL"
bash "$ROOT/scripts/eval_restoration_pilot.sh" --runs "$UPSTREAM" "$OUT/evaluation" "$CARD" "$OUT/formal/${MODEL}_gold_pixels"
"$PY" -m revrender.tools.restoration_report --evaluation "$OUT/evaluation" --expected-models 1 --output-dir "$OUT/report"
touch "$OUT/COMPLETED"
