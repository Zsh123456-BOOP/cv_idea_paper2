#!/usr/bin/env bash
# One RevRender run: smoke_a -> smoke_b exact replay -> learning gate -> formal -> freeze.
#   launch_train_arm.sh VARIANT GPU_UUID NUMA BUNDLE BUNDLE_SHA CORRECTED CORRECTED_SHA OUT [MIN_FREE_MIB]
# VARIANT: control | nofine | nocontext | nodense.  CORRECTED may be '-' for the bundle's own training lines only.
# RECIPE=<toml> trains with another recipe file (default configs/revrender.toml; a tiny one checks the whole chain quickly).
# STOP_AFTER=learning runs the three gates (smoke, replay, learning) and stops: it checks a commit or an environment without
# the formal run.  SYNTHETIC=1 trains on a synthetic set (data.synthetic_set): BUNDLE and BUNDLE_SHA name the set and its
# manifest's sha, MAPPING=<index_manifest.json> names paper 1's splits, CORRECTED must be '-', every validation's weights are
# kept and nothing is frozen (the version is chosen later on the user's validation questions).  With SYNTHETIC=1 also:
# SAME_STUDENT=<pairs.json> (a test student's other id is a test writer), INIT=<checkpoint> (start from its weights), and
# REAL=<bundle> REAL_SHA=<sha> [REAL_SHARE=.5] [REAL_CORRECTED=<dir> REAL_CORRECTED_SHA=<sha>] (mix paper 1's real training lines in).
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
RECIPE_ARGS=(); if [ -n "${RECIPE:-}" ]; then RECIPE_ARGS=(--recipe "$RECIPE"); fi
EXTRA=(); if [ "$CORRECTED" != "-" ]; then EXTRA=(--corrected-bundle "$CORRECTED" --corrected-sha "$CORRECTED_SHA"); fi
DATA=(--bundle "$BUNDLE" --bundle-sha "$BUNDLE_SHA")
if [ -n "${SYNTHETIC:-}" ]; then
    test "$CORRECTED" = - && test -f "${MAPPING:-}" || { echo 'SYNTHETIC needs MAPPING and no corrected rows'; exit 2; }
    DATA=(--synthetic "$BUNDLE" --synthetic-sha "$BUNDLE_SHA" --mapping "$MAPPING" --keep-checkpoints)
    if [ -n "${SAME_STUDENT:-}" ]; then DATA+=(--same-student "$SAME_STUDENT"); fi
    if [ -n "${INIT:-}" ]; then DATA+=(--init "$INIT"); fi
    if [ -n "${REAL:-}" ]; then
        DATA+=(--bundle "$REAL" --bundle-sha "$REAL_SHA" --real-share "${REAL_SHARE:-.5}")
        if [ -n "${REAL_CORRECTED:-}" ]; then DATA+=(--corrected-bundle "$REAL_CORRECTED" --corrected-sha "$REAL_CORRECTED_SHA"); fi
    fi
fi
echo "GPU lock held (shared) $(date -u +%FT%TZ)"
run() {
    local stage="$1"; shift
    numactl --cpunodebind="$NUMA" --membind="$NUMA" "$PY" -m revrender.train --variant "$VARIANT" \
        "${DATA[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
        --backbone-weights "$WEIGHTS/R-50-torchvision.pkl" ${RECIPE_ARGS[@]+"${RECIPE_ARGS[@]}"} "$@" > "$OUT/$stage.log" 2>&1
    test -f "$OUT/$stage/complete.json"
    echo "$stage complete $(date -u +%FT%TZ)"
}
run smoke_a --stage smoke --output-dir "$OUT/smoke_a"
run smoke_b --stage smoke --output-dir "$OUT/smoke_b" --replay-of "$OUT/smoke_a"
run learning --stage learning --output-dir "$OUT/learning"
if [ "${STOP_AFTER:-}" = learning ]; then
    flock -u 9; touch "$OUT/GATES_COMPLETE"; echo "gates complete; GPU released $(date -u +%FT%TZ)"; exit 0
fi
run formal --stage formal --output-dir "$OUT/formal" --gate-root "$OUT"
flock -u 9
echo "training complete; GPU released $(date -u +%FT%TZ)"
if [ -n "${SYNTHETIC:-}" ]; then touch "$OUT/COMPLETE"; echo "checkpoints kept; nothing frozen $(date -u +%FT%TZ)"; exit 0; fi
"$PY" -m revrender.eval.freeze --bundle "$BUNDLE" --bundle-sha "$BUNDLE_SHA" --run "$OUT/formal" --output "$OUT/freeze.json" > "$OUT/freeze.log" 2>&1
touch "$OUT/COMPLETE"
echo "validation frozen; no test pass $(date -u +%FT%TZ)"
