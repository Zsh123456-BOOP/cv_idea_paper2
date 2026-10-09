#!/usr/bin/env bash
# Six published baselines, one sequential lane per authorized card.
# Usage: LANE DATA UPSTREAM OUT [smoke|formal|resume] [PHYSICAL_CARD] [MODELS...]
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
LANE="$1"; DATA="$2"; UPSTREAM="$3"; OUT="$4"; MODE="${5:-formal}"
case "$LANE" in
  0) GPU=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; NUMA=0; MODELS=(COR-Generator Restormer) ;;
  2) GPU=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; NUMA=1; MODELS=(NAFNet DE-GAN) ;;
  3) GPU=GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b; NUMA=1; MODELS=(Strikethrough-CycleGAN DocDiff) ;;
  *) echo 'only lanes 0/2/3 authorized'; exit 2 ;;
esac
CARD="${6:-$LANE}"
[ "$#" -lt 7 ] || MODELS=("${@:7}")
case "$CARD" in
  0) GPU=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; NUMA=0 ;;
  2) GPU=GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0; NUMA=1 ;;
  3) GPU=GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b; NUMA=1 ;;
  *) echo 'only physical cards 0/2/3 authorized'; exit 2 ;;
esac
case "$MODE" in smoke|formal|resume) ;; *) exit 2 ;; esac
require_clean_source
if [ "$MODE" = resume ]; then
  test -d "$OUT" && test ! -f "$OUT/COMPLETED"
  if [ -f "$OUT/FAILED" ]; then mv "$OUT/FAILED" "$OUT/FAILED.$(date -u +%Y%m%dT%H%M%SZ)"; fi
else
  test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
  mkdir "$OUT"
fi
trap 'code=$?; if [ "$code" -ne 0 ]; then touch "$OUT/FAILED"; fi' EXIT
export ALLOW_GPU3=1
EXTRA=()
[ "$MODE" != smoke ] || EXTRA=(--steps 100 --eval-every 100 --validation-limit 4)
for model in "${MODELS[@]}"; do
  dest="$OUT/${model}_gold_pixels"
  if [ "$MODE" = resume ] && [ -f "$dest/selection.json" ] && [ -f "$dest/inference_verified.json" ]; then
    echo "preserving completed model: $dest"
    continue
  fi
  test ! -e "$dest" || { echo "incomplete existing model requires explicit recovery: $dest"; exit 2; }
  test ! -e "$OUT/${model}.log" || { echo 'existing model log will not be overwritten'; exit 2; }
  IDLE=0; [ "$CARD" != 3 ] || IDLE=1
  wait_for_gpu_memory "$GPU" 20000 "$IDLE"
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
  bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" "$NUMA" "$dest" -- \
    "$RUNTIME" -m revrender.baselines.restoration --model "$model" --arm gold_pixels \
    --protocol "$ROOT/configs/restoration.toml" --data "$DATA" --upstream "$UPSTREAM" \
    --human-validation /home/zsh/hme/outputs/paper2_pixel_finetune_set20261007/validation_native.json \
    --output-dir "$dest" "${EXTRA[@]}" > "$OUT/${model}.log" 2>&1
  # Reload the frozen inference checkpoint on validation pixels, not on test images.
  (hold_card "$GPU"; numactl --cpunodebind="$NUMA" --membind="$NUMA" "$RUNTIME" -m revrender.baselines.restoration_verify \
    --run "$dest" --upstream "$UPSTREAM") > "$OUT/${model}.verify.log" 2>&1
  unset LD_LIBRARY_PATH
 done
touch "$OUT/COMPLETED"
