#!/usr/bin/env bash
# Evaluate only frozen selections, after both training queues finish.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
RUNS=()
GPU=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; CARD=0; NUMA=0
if [ "$1" = --runs ]; then
  UPSTREAM="$2"; OUT="$3"; CARD="$4"; RUNS=("${@:5}")
  test "${#RUNS[@]}" -gt 0
  for run in "${RUNS[@]}"; do test -f "$run/inference_verified.json"; done
elif [ "$1" = --suite ]; then
  SUITE="$2"; UPSTREAM="$3"; OUT="$4"
  for lane in 0 2 3; do test -f "$SUITE/lane$lane/COMPLETED"; done
  RUNS=("$SUITE/lane0/COR-Generator_gold_pixels" "$SUITE/lane0/Restormer_gold_pixels"
        "$SUITE/lane2/NAFNet_gold_pixels" "$SUITE/lane2/DE-GAN_gold_pixels"
        "$SUITE/lane3/Strikethrough-CycleGAN_gold_pixels" "$SUITE/lane3/DocDiff_gold_pixels")
  CARD="${5:-3}"
else
  LANE0="$1"; LANE3="$2"; UPSTREAM="$3"; OUT="$4"
  test -f "$LANE0/COMPLETED" && test -f "$LANE3/COMPLETED"
  for arm in gold_pixels predicted_pixels predicted_traced; do
    RUNS+=("$LANE0/NAFNet_$arm" "$LANE0/DE-GAN_$arm" "$LANE3/DocDiff_$arm")
  done
fi
case "$CARD" in
  0) GPU=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; NUMA=0 ;;
  2) GPU=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; NUMA=1 ;;
  3) GPU=GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b; NUMA=1 ;;
  *) exit 2 ;;
esac
export ALLOW_GPU3=1
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir "$OUT"
trap 'code=$?; if [ "$code" -ne 0 ]; then touch "$OUT/FAILED"; fi' EXIT
IDLE=0; [ "$CARD" != 3 ] || IDLE=1
O=/home/zsh/hme/outputs
LABELS=("$O"/paper2_annotation_labels/*revision-boxes-test_*.json)
test "${#LABELS[@]}" = 1
EXPORTS=()
for run in "${RUNS[@]}"; do
    name="$(basename "$run")"
    model="${name%_gold_pixels}"
    case "$name" in DE-GAN_*) model=DE-GAN ;; esac
    test -f "$run/selection.json"
    RUNTIME="$PY"
    if [ "$model" = DE-GAN ]; then
      RUNTIME=/home/zsh/hme/envs/paper2_restoration_tf/bin/python
      export TF_USE_LEGACY_KERAS=1 LD_LIBRARY_PATH=""
      for directory in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do
        export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      done
    else
      unset LD_LIBRARY_PATH
    fi
    dest="$OUT/$name"
    wait_for_gpu_memory "$GPU" 18000 "$IDLE"
    bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" "$NUMA" "$dest" -- \
      "$RUNTIME" -m revrender.baselines.restoration_export export --run "$run" --upstream "$UPSTREAM" \
      --labels "${LABELS[0]}" --selection "$O/paper2_annotation_labels/selection.json" \
      --geometry "$O/paper2_geometry_labels_test7" \
      --pages "$O/paper2_pages_clean_unlabeled3976" --pages /home/zsh/hme/data/full1026_v1/images \
      --output-dir "$dest" > "$dest.log" 2>&1
    EXPORTS+=("$dest")
done
unset LD_LIBRARY_PATH
PYTHONPATH="$ROOT" "$PY" -m revrender.baselines.restoration_export merge --inputs "${EXPORTS[@]}" --output-dir "$OUT/pack"
OCR_MEMORY=.9; OCR_FREE=22500
if [ "$CARD" = 2 ]; then OCR_MEMORY=.85; OCR_FREE=21000; fi
wait_for_gpu_memory "$GPU" "$OCR_FREE" "$IDLE"
# vLLM resolves logical CUDA ordinals; hold the UUID lock, then expose the corresponding physical ordinal.
bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" "$NUMA" "$OUT/qwen8b" -- env CUDA_VISIBLE_DEVICES="$CARD" \
  "$PY" -m revrender.recognize_lines --pack "$OUT/pack" --model /home/zsh/hme/models/Qwen3-VL-8B-Instruct \
  --conditions erased_neutral --gpu-memory "$OCR_MEMORY" --output-dir "$OUT/qwen8b" > "$OUT/qwen8b.log" 2>&1
PYTHONPATH="$ROOT" "$PY" -m revrender.eval.revision_boxes read --pack "$OUT/pack" \
  --readings "$OUT/qwen8b/predictions.jsonl" --output "$OUT/read_qwen8b.json"
touch "$OUT/COMPLETED"
