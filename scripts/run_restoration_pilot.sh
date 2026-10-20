#!/usr/bin/env bash
# Three architectures x three material arms. Only user-authorized cards 0/3.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
LANE="$1"; DATA="$2"; UPSTREAM="$3"; OUT="$4"; MODE="${5:-formal}"
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir "$OUT"
case "$LANE" in
  0) GPU=GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0; NUMA=0; MODELS=(NAFNet DE-GAN) ;;
  3) GPU=GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b; NUMA=1; MODELS=(DocDiff) ;;
  *) echo 'only lanes 0/3 are authorized'; exit 2 ;;
esac
export ALLOW_GPU3=1
ARMS=(gold_pixels predicted_pixels predicted_traced)
STEPS=4000; VALIDATION=64; EVAL=1000
if [ "$MODE" = smoke ]; then ARMS=(gold_pixels); STEPS=3; VALIDATION=4; EVAL=3; fi
for arm in "${ARMS[@]}"; do
  for model in "${MODELS[@]}"; do
    require_free_memory "$GPU" 18000
    RUNTIME="$PY"
    if [ "$model" = DE-GAN ]; then
      RUNTIME=/home/zsh/hme/envs/paper2_restoration_tf/bin/python
      export TF_USE_LEGACY_KERAS=1
      export LD_LIBRARY_PATH=""
      for directory in /home/zsh/hme/envs/paper2_restoration_tf/lib/python3.12/site-packages/nvidia/*/lib; do
        export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      done
    else
      unset LD_LIBRARY_PATH
    fi
    bash "$ROOT/scripts/launch_gpu_job.sh" "$GPU" "$NUMA" "$OUT/${model}_${arm}" -- \
      "$RUNTIME" -m revrender.baselines.restoration --model "$model" --arm "$arm" \
      --data "$DATA" --upstream "$UPSTREAM" \
      --human-validation /home/zsh/hme/outputs/paper2_fixed_validation20261007/validation_native.json \
      --output-dir "$OUT/${model}_${arm}" --steps "$STEPS" --batch-size 4 \
      --eval-every "$EVAL" --validation-limit "$VALIDATION" > "$OUT/${model}_${arm}.log" 2>&1
  done
done
touch "$OUT/COMPLETED"
