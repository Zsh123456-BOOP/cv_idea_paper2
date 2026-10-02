#!/usr/bin/env bash
# One native baseline, seed 42: 4-step smoke -> formal -> freeze at the upstream threshold -> mask-threshold freeze.
#   launch_native_arm.sh MODEL GPU_UUID NUMA BUNDLE BUNDLE_SHA CORRECTED CORRECTED_SHA OUT MIN_FREE [-- extra train arguments]
# MODEL: maskdino|pointrend|mask2former|condinst|maskrcnn|solov2|dimaskdino.  CORRECTED may be '-' (bundle lines only).
# Mask Transfiner: MODEL=maskrcnn and TRANSFINER_PIN=<pin file>.  Journal arms: extra arguments such as --fraction 0.5,
# --dense, --data synthetic --synthetic-root DIR go after `--`.  STOP_AFTER=smoke runs the 4-step smoke and stops.
# Holds the card's lock shared; OUT must not exist.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
MODEL="$1"; GPU_UUID="$2"; NUMA="$3"; BUNDLE="$4"; BUNDLE_SHA="$5"; CORRECTED="$6"; CORRECTED_SHA="$7"; OUT="$8"; MIN_FREE="${9:-12000}"; shift $(( $# < 9 ? $# : 9 ))
if [ "$#" -gt 0 ]; then test "$1" = -- || { echo 'usage: ... MIN_FREE -- extra train arguments'; exit 2; }; shift; fi
case "$MODEL" in maskdino|pointrend|mask2former|condinst|maskrcnn|solov2|dimaskdino) ;; *) echo 'undeclared model'; exit 2 ;; esac
refuse_card3 "$GPU_UUID"
UPSTREAM="$UPSTREAM_V1"; case "$MODEL" in solov2|dimaskdino) UPSTREAM="$UPSTREAM_V2" ;; esac
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir -p "$OUT"
trap 'echo "launcher failed at line $LINENO $(date -u +%FT%TZ)"; touch "$OUT/FAILED"' ERR
hold_card "$GPU_UUID"
require_free_memory "$GPU_UUID" "$MIN_FREE"
export HME_NUMA_NODE="$NUMA"
ARGS=("$@")
TF=(); if [ -n "${TRANSFINER_PIN:-}" ]; then TF=(--transfiner-pin "$TRANSFINER_PIN"); fi
EXTRA=(); if [ "$CORRECTED" != "-" ]; then EXTRA=(--corrected-bundle "$CORRECTED" --corrected-sha "$CORRECTED_SHA"); fi
echo "GPU lock held (shared) $(date -u +%FT%TZ): $MODEL ${ARGS[*]-}"
run_native() {
    local stage="$1"; shift
    numactl --cpunodebind="$NUMA" --membind="$NUMA" "$NATIVE_PY" -m revrender.baselines.native.train --method "$MODEL" \
        --bundle "$BUNDLE" --bundle-sha "$BUNDLE_SHA" ${EXTRA[@]+"${EXTRA[@]}"} ${TF[@]+"${TF[@]}"} \
        --upstream-root "$UPSTREAM" --weights-dir "$WEIGHTS" --microbatch 2 --output-dir "$OUT/$stage" \
        ${ARGS[@]+"${ARGS[@]}"} "$@" > "$OUT/$stage.log" 2>&1
    echo "$stage complete $(date -u +%FT%TZ)"
}
run_native smoke --smoke-steps 4
test -f "$OUT/smoke/smoke_complete.json"
if [ "${STOP_AFTER:-}" = smoke ]; then
    flock -u 9; touch "$OUT/SMOKE_COMPLETE"; echo "smoke complete; GPU released $(date -u +%FT%TZ)"; exit 0
fi
run_native formal
test -f "$OUT/formal/train_complete.json"
CUDA_VISIBLE_DEVICES= "$PY" -m revrender.eval.freeze --model "$MODEL" --bundle "$BUNDLE" --bundle-sha "$BUNDLE_SHA" \
    --run "$OUT/formal" --output "$OUT/freeze_upstream.json" > "$OUT/freeze_upstream.log" 2>&1
numactl --cpunodebind="$NUMA" --membind="$NUMA" "$NATIVE_PY" -m revrender.baselines.native.threshold --method "$MODEL" \
    --freeze "$OUT/freeze_upstream.json" --bundle "$BUNDLE" --bundle-sha "$BUNDLE_SHA" ${TF[@]+"${TF[@]}"} \
    --upstream-root "$UPSTREAM" --weights-dir "$WEIGHTS" --output-dir "$OUT/threshold" > "$OUT/threshold.log" 2>&1
cp "$OUT/threshold/freeze.json" "$OUT/freeze.json"
flock -u 9
touch "$OUT/COMPLETE"
echo "trained and frozen on validation; no test pass $(date -u +%FT%TZ)"
