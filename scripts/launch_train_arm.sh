#!/usr/bin/env bash
# One RevRender run: smoke_a -> smoke_b exact replay -> learning gate -> formal -> freeze.
#   launch_train_arm.sh VARIANT GPU_UUID NUMA BUNDLE BUNDLE_SHA CORRECTED CORRECTED_SHA OUT [MIN_FREE_MIB]
# VARIANT: control | nofine | nocontext | nodense.  CORRECTED may be '-' for the bundle's own training lines only.
# Holds the card's lock shared; OUT must not exist.  No test split is read.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
VARIANT="$1"; GPU_UUID="$2"; NUMA="$3"; BUNDLE="$4"; BUNDLE_SHA="$5"; CORRECTED="$6"; CORRECTED_SHA="$7"; OUT="$8"; MIN_FREE="${9:-8000}"
case "$VARIANT" in control|nofine|nocontext|nodense) ;; *) echo "undeclared variant"; exit 2 ;; esac
refuse_card3 "$GPU_UUID"
test ! -e "$OUT" || { echo 'create-only output exists'; exit 2; }
require_clean_source
mkdir -p "$OUT"
trap 'echo "launcher failed at line $LINENO $(date -u +%FT%TZ)"; touch "$OUT/FAILED"' ERR
hold_card "$GPU_UUID"
require_free_memory "$GPU_UUID" "$MIN_FREE"
export HME_NUMA_NODE="$NUMA"
EXTRA=(); if [ "$CORRECTED" != "-" ]; then EXTRA=(--corrected-bundle "$CORRECTED" --corrected-sha "$CORRECTED_SHA"); fi
echo "GPU lock held (shared) $(date -u +%FT%TZ)"
run() {
    local stage="$1"; shift
    numactl --cpunodebind="$NUMA" --membind="$NUMA" "$PY" -m revrender.train --variant "$VARIANT" \
        --bundle "$BUNDLE" --bundle-sha "$BUNDLE_SHA" ${EXTRA[@]+"${EXTRA[@]}"} \
        --backbone-weights "$WEIGHTS/R-50-torchvision.pkl" "$@" > "$OUT/$stage.log" 2>&1
    test -f "$OUT/$stage/complete.json"
    echo "$stage complete $(date -u +%FT%TZ)"
}
run smoke_a --stage smoke --output-dir "$OUT/smoke_a"
run smoke_b --stage smoke --output-dir "$OUT/smoke_b" --replay-of "$OUT/smoke_a"
run learning --stage learning --output-dir "$OUT/learning"
run formal --stage formal --output-dir "$OUT/formal" --gate-root "$OUT"
flock -u 9
echo "training complete; GPU released $(date -u +%FT%TZ)"
"$PY" -m revrender.eval.freeze --bundle "$BUNDLE" --bundle-sha "$BUNDLE_SHA" --run "$OUT/formal" --output "$OUT/freeze.json" > "$OUT/freeze.log" 2>&1
touch "$OUT/COMPLETE"
echo "validation frozen; no test pass $(date -u +%FT%TZ)"
